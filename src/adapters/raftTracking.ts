// @ts-nocheck
import { getCapabilities } from './util'
import { ensureModel } from './cache'

/**
 * Memory monitoring utility
 */
function logMemoryUsage(label: string) {
  if (typeof performance !== 'undefined' && performance.memory) {
    const mem = performance.memory
    console.log(`[MEMORY] ${label}:`, {
      'Used Heap': `${(mem.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
      'Total Heap': `${(mem.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
      'Heap Limit': `${(mem.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`,
      'Usage %': `${((mem.usedJSHeapSize / mem.jsHeapSizeLimit) * 100).toFixed(1)}%`,
    })
  } else {
    console.log(`[MEMORY] ${label}: Memory API not available (use Chrome with --enable-precise-memory-info)`)
  }
}

function calculateTensorMemory(dims: number[], type: string = 'float32'): number {
  const totalElements = dims.reduce((a, b) => a * b, 1)
  const bytesPerElement = type === 'float32' ? 4 : type === 'int8' ? 1 : 4
  return totalElements * bytesPerElement
}

/**
 * RAFT-based optical flow tracking for mask propagation
 * Uses ONNX Runtime with WebGPU acceleration
 */
export class RAFTMaskTracker {
  private raftSession: ort.InferenceSession | null = null
  private previousFrame: ImageData | null = null
  private currentMask: ImageData | null = null

  async initialize() {
    console.log('[RAFTMaskTracker] Initializing RAFT model...')
    console.log('═══════════════════════════════════════════════')
    console.log('📊 MEMORY DIAGNOSIS MODE ENABLED')
    console.log('═══════════════════════════════════════════════')
    logMemoryUsage('Before RAFT initialization')

    try {
      // Ensure ort is available
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime not loaded. Please ensure the script is loaded first.')
      }

      // Set wasm paths to CDN
      ort.env.wasm.wasmPaths =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/'

      // Configure WASM settings for RAFT model (large 48MB model needs optimal memory)
      if (typeof navigator !== 'undefined') {
        // Enable multi-threading if SharedArrayBuffer is available (better memory handling)
        if (typeof SharedArrayBuffer !== 'undefined') {
          ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4)
          console.log(`[RAFTMaskTracker] Enabling ${ort.env.wasm.numThreads} WASM threads for better memory management`)
        } else {
          ort.env.wasm.numThreads = 1
          console.log('[RAFTMaskTracker] Single-threaded mode (SharedArrayBuffer not available)')
        }
        
        if (
          typeof WebAssembly !== 'undefined' &&
          typeof WebAssembly.validate === 'function'
        ) {
          ort.env.wasm.simd = true
        }
        ort.env.wasm.proxy = true // Enable proxy for better stability with large models
        // @ts-ignore - these properties exist in ort.env.wasm
        ort.env.wasm.initTimeout = 60000 // 60 seconds for large model
      }

      // Configure WebGPU settings
      if (typeof ort.env.webgpu !== 'undefined') {
        console.log('[RAFTMaskTracker] Configuring WebGPU environment')
        ort.env.webgpu.powerPreference = 'high-performance'
        ort.env.webgpu.forceFallbackAdapter = false
      }

      console.log('[RAFTMaskTracker] WASM config:', ort.env.wasm)

      const capabilities = await getCapabilities()
      console.log('[RAFTMaskTracker] Capabilities:', capabilities)
      console.log(
        '[RAFTMaskTracker] WebGPU available:',
        capabilities.webgpu
      )
      console.log('[RAFTMaskTracker] WASM available:', capabilities.wasm)
      console.log('[RAFTMaskTracker] SIMD available:', capabilities.simd)
      console.log('[RAFTMaskTracker] Threads available:', capabilities.threads)

      const modelBuffer = await ensureModel('raft')
      console.log(
        '[RAFTMaskTracker] Model loaded, size:',
        `${(modelBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`
      )
      logMemoryUsage('After model buffer loaded')

      // Clone the buffer to prevent detachment issues
      const cloneBuffer = (buffer: ArrayBuffer) => {
        const copy = new ArrayBuffer(buffer.byteLength)
        new Uint8Array(copy).set(new Uint8Array(buffer))
        return copy
      }

      // Try different execution providers with fallback
      let session = null
      let selectedProvider = null
      const providersToTry = []

      // WebGPU doesn't support DequantizeLinear operator used in quantized RAFT model
      // Force WASM for full operator compatibility
      console.log(
        '[RAFTMaskTracker] ⚠️  Using WASM only - RAFT model uses quantized weights (DequantizeLinear) not supported by WebGPU'
      )
      providersToTry.push({ name: 'wasm', providers: ['wasm'] })

      console.log(
        '[RAFTMaskTracker] Provider priority order:',
        providersToTry.map((p) => p.name)
      )

      for (const { name, providers } of providersToTry) {
        try {
          console.log(
            `[RAFTMaskTracker] 🔧 Attempting to create session with ${name}...`
          )
          // Clone buffer for each attempt to prevent detachment
          const bufferCopy = cloneBuffer(modelBuffer)

          // Session options - keep simple for maximum compatibility
          let sessionOptions: any = {
            executionProviders: providers,
          }

          // Add backend-specific optimizations
          if (name === 'webgpu') {
            console.log('[RAFTMaskTracker] Configuring WebGPU session...')
            // Keep WebGPU options minimal - complex options can cause failures
            sessionOptions = {
              executionProviders: ['webgpu'],
            }
          } else if (name === 'wasm') {
            console.log('[RAFTMaskTracker] Configuring WASM session for large model...')
            // Optimized for large models (48MB RAFT)
            sessionOptions = {
              executionProviders: ['wasm'],
              graphOptimizationLevel: 'extended', // Better optimization for large models
              executionMode: 'sequential', // Sequential execution saves memory
              enableCpuMemArena: false, // Don't pre-allocate large memory arenas
              enableMemPattern: false, // Disable memory pattern for large models
            }
            console.log('[RAFTMaskTracker] Using memory-optimized WASM configuration')
          }

          console.log('[RAFTMaskTracker] Session options:', JSON.stringify(sessionOptions))

          console.log(`[RAFTMaskTracker] 🔨 Creating ONNX session with ${name}...`)
          logMemoryUsage(`Before session creation (${name})`)
          
          const sessionStart = performance.now()
          session = await ort.InferenceSession.create(
            bufferCopy,
            sessionOptions
          )
          const sessionTime = ((performance.now() - sessionStart) / 1000).toFixed(2)
          
          logMemoryUsage(`After session creation (${name})`)
          console.log(`[RAFTMaskTracker] Session created in ${sessionTime}s`)
          console.log('[RAFTMaskTracker] Session inputs:', session.inputNames)
          console.log('[RAFTMaskTracker] Session outputs:', session.outputNames)
          
          selectedProvider = name
          console.log(
            `[RAFTMaskTracker] ✅ Successfully created session with ${name.toUpperCase()}`
          )
          break
        } catch (err) {
          const errorMsg = err?.message || String(err)
          console.error(
            `[RAFTMaskTracker] ❌ Failed to create session with ${name}:`
          )
          console.error('Error details:', err)
          console.error('Error type:', typeof err)
          console.error('Error string:', String(err))

          // Log WebGPU-specific failures with details
          if (name === 'webgpu') {
            console.error(
              '⚠️  [RAFTMaskTracker] WebGPU initialization failed!'
            )
            console.error('WebGPU Error:', errorMsg)
            console.error(
              'Falling back to WASM (may be slower and use more memory)...'
            )
          }

          if (providersToTry[providersToTry.length - 1].name === name) {
            throw new Error(
              `Failed to initialize RAFT model with any execution provider. Last error: ${errorMsg}`
            )
          }
        }
      }

      if (!session) {
        throw new Error('Failed to create RAFT inference session')
      }

      this.raftSession = session

      console.log('═══════════════════════════════════════════════')
      console.log('🎉 [RAFTMaskTracker] RAFT model loaded successfully')
      console.log(`🚀 [RAFTMaskTracker] Using execution provider: ${selectedProvider.toUpperCase()}`)
      console.log('═══════════════════════════════════════════════')
      console.log('[RAFTMaskTracker] Input names:', this.raftSession.inputNames)
      console.log(
        '[RAFTMaskTracker] Output names:',
        this.raftSession.outputNames
      )
    } catch (error) {
      console.error('[RAFTMaskTracker] Error initializing RAFT:', error)
      console.error('[RAFTMaskTracker] Error type:', typeof error)
      console.error('[RAFTMaskTracker] Error name:', error?.name)
      console.error('[RAFTMaskTracker] Error message:', error?.message)
      console.error('[RAFTMaskTracker] Error code:', error?.code)
      throw error
    }
  }

  /**
   * Set initial frame and mask for tracking
   */
  setReference(frame: HTMLImageElement, mask: HTMLCanvasElement) {
    console.log('[RAFTMaskTracker] Setting reference frame and mask')
    this.previousFrame = this.imageToImageData(frame)
    this.currentMask = this.canvasToImageData(mask)
  }

  /**
   * Track mask to next frame using optical flow
   */
  async track(nextFrame: HTMLImageElement): Promise<HTMLCanvasElement> {
    if (!this.raftSession || !this.previousFrame || !this.currentMask) {
      throw new Error('RAFT tracker not initialized or reference not set')
    }

    try {
      console.log('[RAFTMaskTracker] Computing optical flow and warping mask')

      const nextFrameData = this.imageToImageData(nextFrame)

      // Compute optical flow from previous to next frame
      const { flow, flowWidth, flowHeight } = await this.computeOpticalFlow(
        this.previousFrame,
        nextFrameData
      )

      // Warp mask using computed flow
      const warpedMask = this.warpMask(
        this.currentMask,
        flow,
        flowWidth,
        flowHeight
      )

      // Update state for next iteration
      this.previousFrame = nextFrameData
      this.currentMask = warpedMask

      // Convert to canvas
      return this.imageDataToCanvas(warpedMask)
    } catch (error) {
      console.error('[RAFTMaskTracker] Error in track():', error)
      throw error
    }
  }

  /**
   * Compute optical flow using RAFT model
   */
  private async computeOpticalFlow(
    frame1: ImageData,
    frame2: ImageData
  ): Promise<{ flow: Float32Array; flowWidth: number; flowHeight: number }> {
    if (!this.raftSession) {
      throw new Error('RAFT session not initialized')
    }

    try {
      // Prepare tensors
      console.log('─────────────────────────────────────────────')
      console.log('📊 INFERENCE MEMORY DIAGNOSIS')
      console.log('─────────────────────────────────────────────')
      logMemoryUsage('Before tensor creation')
      
      console.log('[RAFTMaskTracker] Preparing input tensors...')
      console.log(
        '[RAFTMaskTracker] Frame dimensions:',
        frame1.width,
        'x',
        frame1.height
      )

      const frame1Tensor = this.imageDataToTensor(frame1)
      const frame2Tensor = this.imageDataToTensor(frame2)

      console.log('[RAFTMaskTracker] Frame1 tensor shape:', frame1Tensor.dims)
      console.log('[RAFTMaskTracker] Frame2 tensor shape:', frame2Tensor.dims)

      // Calculate memory usage of tensors
      const tensor1Memory = calculateTensorMemory(frame1Tensor.dims, 'float32')
      const tensor2Memory = calculateTensorMemory(frame2Tensor.dims, 'float32')
      console.log(`[RAFTMaskTracker] 📊 Tensor1 memory: ${(tensor1Memory / 1024 / 1024).toFixed(2)} MB`)
      console.log(`[RAFTMaskTracker] 📊 Tensor2 memory: ${(tensor2Memory / 1024 / 1024).toFixed(2)} MB`)
      console.log(`[RAFTMaskTracker] 📊 Total input memory: ${((tensor1Memory + tensor2Memory) / 1024 / 1024).toFixed(2)} MB`)
      
      logMemoryUsage('After tensor creation')

      // Validate tensor data
      const data1 = frame1Tensor.data as Float32Array
      const data2 = frame2Tensor.data as Float32Array
      console.log('[RAFTMaskTracker] Tensor1 data size:', data1.length, 'elements')
      console.log('[RAFTMaskTracker] Tensor2 data size:', data2.length, 'elements')
      console.log('[RAFTMaskTracker] Tensor1 sample values:', [
        data1[0],
        data1[100],
        data1[1000],
      ])
      console.log('[RAFTMaskTracker] Tensor2 sample values:', [
        data2[0],
        data2[100],
        data2[1000],
      ])

      // Check for NaN or Infinity
      const hasInvalidData1 =
        data1.some((v) => !isFinite(v)) || data1.length === 0
      const hasInvalidData2 =
        data2.some((v) => !isFinite(v)) || data2.length === 0

      if (hasInvalidData1 || hasInvalidData2) {
        throw new Error(
          `Invalid tensor data detected: tensor1=${hasInvalidData1}, tensor2=${hasInvalidData2}`
        )
      }

      console.log('[RAFTMaskTracker] Running RAFT inference...')

      // Use the actual input names from the model
      const inputNames = this.raftSession.inputNames
      const feeds = {
        [inputNames[0]]: frame1Tensor,
        [inputNames[1]]: frame2Tensor,
      }

      console.log('[RAFTMaskTracker] Input feeds:', Object.keys(feeds))

      console.log('\n🔍 ===== MEMORY BEFORE INFERENCE =====')
      logMemoryUsage('Before session.run()')
      console.log('=======================================\n')

      const startTime = performance.now()
      let results
      try {
        results = await this.raftSession.run(feeds)
      } catch (runError) {
        console.log('\n❌ ===== MEMORY AT INFERENCE FAILURE =====')
        logMemoryUsage('At error')
        console.log('==========================================\n')
        
        // Handle WASM numeric errors
        const errorMessage =
          typeof runError === 'number'
            ? `WASM error code: ${runError} (likely out of memory or invalid input)`
            : runError.message || String(runError)
        throw new Error(`RAFT inference failed: ${errorMessage}`)
      }
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2)

      console.log('\n✅ ===== MEMORY AFTER INFERENCE SUCCESS =====')
      logMemoryUsage('After session.run()')
      console.log('============================================\n')

      console.log(`[RAFTMaskTracker] RAFT inference completed in ${elapsed}s`)
      console.log('[RAFTMaskTracker] Output keys:', Object.keys(results))

      // Get the flow output (use first output if 'flow' doesn't exist)
      const outputName = this.raftSession.outputNames[0]
      const flowTensor = results[outputName]

      if (!flowTensor) {
        throw new Error(`Expected output '${outputName}' not found in results`)
      }

      console.log('[RAFTMaskTracker] Flow tensor shape:', flowTensor.dims)
      console.log('[RAFTMaskTracker] Flow tensor type:', flowTensor.type)

      // Flow tensor should be [1, 2, H, W] format
      const flowHeight = flowTensor.dims[2]
      const flowWidth = flowTensor.dims[3]

      return {
        flow: flowTensor.data as Float32Array,
        flowWidth,
        flowHeight,
      }
    } catch (error) {
      console.error('[RAFTMaskTracker] Error computing optical flow:', error)
      console.error('[RAFTMaskTracker] Error details:', {
        message: error.message,
        stack: error.stack,
        code: error.code,
      })
      throw error
    }
  }

  /**
   * Warp mask using optical flow field
   */
  private warpMask(
    mask: ImageData,
    flow: Float32Array,
    flowWidth: number,
    flowHeight: number
  ): ImageData {
    const { width: origWidth, height: origHeight } = mask

    console.log(
      `[RAFTMaskTracker] Warping mask (${origWidth}x${origHeight}) using flow (${flowWidth}x${flowHeight})`
    )

    // Resize mask to match flow dimensions if they differ
    let processedMask = mask
    if (origWidth !== flowWidth || origHeight !== flowHeight) {
      processedMask = this.resizeImageData(mask, flowWidth, flowHeight)
      console.log(
        `[RAFTMaskTracker] Resized mask to ${flowWidth}x${flowHeight} for warping`
      )
    }

    const warpedMask = new ImageData(flowWidth, flowHeight)

    // For each pixel in the new mask
    for (let y = 0; y < flowHeight; y++) {
      for (let x = 0; x < flowWidth; x++) {
        const flowIdx = y * flowWidth + x

        // Flow tensor is in [2, H, W] format, so we need to access channels properly
        // Channel 0 = dx, Channel 1 = dy
        const dx = flow[flowIdx]
        const dy = flow[flowWidth * flowHeight + flowIdx]

        // Source pixel location in previous mask
        const srcX = x - dx
        const srcY = y - dy

        // Bilinear interpolation from source mask
        const maskValue = this.bilinearSample(processedMask, srcX, srcY)

        // Set warped mask value
        const outIdx = (y * flowWidth + x) * 4
        warpedMask.data[outIdx] = maskValue
        warpedMask.data[outIdx + 1] = maskValue
        warpedMask.data[outIdx + 2] = maskValue
        warpedMask.data[outIdx + 3] = 255
      }
    }

    // Resize warped mask back to original dimensions if needed
    if (origWidth !== flowWidth || origHeight !== flowHeight) {
      const resizedWarped = this.resizeImageData(
        warpedMask,
        origWidth,
        origHeight
      )
      console.log(
        `[RAFTMaskTracker] Resized warped mask back to ${origWidth}x${origHeight}`
      )
      return resizedWarped
    }

    return warpedMask
  }

  /**
   * Bilinear interpolation for smooth mask warping
   */
  private bilinearSample(imageData: ImageData, x: number, y: number): number {
    const { width, height, data } = imageData

    // Clamp to image bounds
    x = Math.max(0, Math.min(width - 1, x))
    y = Math.max(0, Math.min(height - 1, y))

    const x0 = Math.floor(x)
    const x1 = Math.min(x0 + 1, width - 1)
    const y0 = Math.floor(y)
    const y1 = Math.min(y0 + 1, height - 1)

    const fx = x - x0
    const fy = y - y0

    // Get values at four corners
    const v00 = data[(y0 * width + x0) * 4]
    const v10 = data[(y0 * width + x1) * 4]
    const v01 = data[(y1 * width + x0) * 4]
    const v11 = data[(y1 * width + x1) * 4]

    // Bilinear interpolation
    const v0 = v00 * (1 - fx) + v10 * fx
    const v1 = v01 * (1 - fx) + v11 * fx
    return v0 * (1 - fy) + v1 * fy
  }

  /**
   * Convert HTMLImageElement to ImageData
   */
  private imageToImageData(img: HTMLImageElement): ImageData {
    const canvas = document.createElement('canvas')
    canvas.width = img.width
    canvas.height = img.height
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    return ctx.getImageData(0, 0, img.width, img.height)
  }

  /**
   * Convert HTMLCanvasElement to ImageData
   */
  private canvasToImageData(canvas: HTMLCanvasElement): ImageData {
    const ctx = canvas.getContext('2d')!
    return ctx.getImageData(0, 0, canvas.width, canvas.height)
  }

  /**
   * Resize ImageData to target dimensions
   */
  private resizeImageData(
    imageData: ImageData,
    targetWidth: number,
    targetHeight: number
  ): ImageData {
    const canvas = document.createElement('canvas')
    canvas.width = imageData.width
    canvas.height = imageData.height
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(imageData, 0, 0)

    const resizedCanvas = document.createElement('canvas')
    resizedCanvas.width = targetWidth
    resizedCanvas.height = targetHeight
    const resizedCtx = resizedCanvas.getContext('2d')!
    resizedCtx.drawImage(canvas, 0, 0, targetWidth, targetHeight)

    return resizedCtx.getImageData(0, 0, targetWidth, targetHeight)
  }

  /**
   * Convert ImageData to ONNX tensor (normalized RGB)
   * RAFT model requires fixed input size of 360x480 (model requirement)
   */
  private imageDataToTensor(imageData: ImageData): ort.Tensor {
    const { width, height, data } = imageData

    // RAFT model expects these exact dimensions - cannot be changed
    const targetWidth = 480
    const targetHeight = 360

    console.log(
      `[RAFTMaskTracker] Original size: ${width}x${height}, Target size: ${targetWidth}x${targetHeight}`
    )

    // Resize if dimensions don't match
    let processedData = imageData
    if (width !== targetWidth || height !== targetHeight) {
      processedData = this.resizeImageData(imageData, targetWidth, targetHeight)
      console.log(
        `[RAFTMaskTracker] Resized to ${processedData.width}x${processedData.height}`
      )
    }

    // Validate processedData
    if (
      !processedData ||
      !processedData.data ||
      processedData.data.length === 0
    ) {
      throw new Error('Invalid image data after resize')
    }

    const expectedSize = targetHeight * targetWidth * 4
    if (processedData.data.length !== expectedSize) {
      throw new Error(
        `Invalid image data size: expected ${expectedSize}, got ${processedData.data.length}`
      )
    }

    const float32Data = new Float32Array(3 * targetHeight * targetWidth)

    // Validate Float32Array was created successfully
    if (!float32Data || float32Data.length === 0) {
      throw new Error('Failed to allocate tensor memory')
    }

    for (let i = 0; i < targetHeight * targetWidth; i++) {
      // Normalize to [0, 1] and convert to CHW format
      const r = processedData.data[i * 4] / 255.0
      const g = processedData.data[i * 4 + 1] / 255.0
      const b = processedData.data[i * 4 + 2] / 255.0

      // Sanity check
      if (!isFinite(r) || !isFinite(g) || !isFinite(b)) {
        throw new Error(`Invalid pixel value at index ${i}`)
      }

      float32Data[i] = r // R
      float32Data[targetHeight * targetWidth + i] = g // G
      float32Data[targetHeight * targetWidth * 2 + i] = b // B
    }

    return new ort.Tensor('float32', float32Data, [
      1,
      3,
      targetHeight,
      targetWidth,
    ])
  }

  /**
   * Convert ImageData to HTMLCanvasElement
   */
  private imageDataToCanvas(imageData: ImageData): HTMLCanvasElement {
    const canvas = document.createElement('canvas')
    canvas.width = imageData.width
    canvas.height = imageData.height
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(imageData, 0, 0)
    return canvas
  }

  cleanup() {
    console.log('[RAFTMaskTracker] Cleaning up resources')
    this.previousFrame = null
    this.currentMask = null
  }
}
