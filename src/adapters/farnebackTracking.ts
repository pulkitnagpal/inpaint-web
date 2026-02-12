// @ts-nocheck
import cv from 'opencv-ts'

/**
 * Farneback optical flow tracking for mask propagation
 * Uses OpenCV.js dense optical flow (no ML model required)
 * More reliable than RAFT for browser environments, though slightly slower
 */
export class FarnebackMaskTracker {
  private previousFrame: any = null // cv.Mat
  private previousFrameGray: any = null // cv.Mat
  private currentMask: ImageData | null = null

  async initialize() {
    console.log('[FarnebackMaskTracker] Initializing Farneback optical flow tracker...')
    console.log('[FarnebackMaskTracker] Using OpenCV.js for dense optical flow computation')
    
    // Verify OpenCV is loaded
    if (typeof cv === 'undefined' || !cv.calcOpticalFlowFarneback) {
      throw new Error('OpenCV.js not loaded. Cannot use Farneback optical flow.')
    }
    
    console.log('[FarnebackMaskTracker] ✅ Farneback tracker ready')
  }

  /**
   * Set initial frame and mask for tracking
   */
  setReference(frame: HTMLImageElement, mask: HTMLCanvasElement) {
    console.log('[FarnebackMaskTracker] Setting reference frame and mask')
    
    // Convert frame to cv.Mat (grayscale for optical flow)
    this.previousFrame = cv.imread(frame)
    this.previousFrameGray = new cv.Mat()
    cv.cvtColor(this.previousFrame, this.previousFrameGray, cv.COLOR_RGBA2GRAY)
    
    // Store mask as ImageData
    this.currentMask = this.canvasToImageData(mask)
    
    console.log('[FarnebackMaskTracker] Reference set:', {
      frameSize: `${this.previousFrame.cols}x${this.previousFrame.rows}`,
      maskSize: `${this.currentMask.width}x${this.currentMask.height}`,
    })
  }

  /**
   * Track mask to next frame using optical flow
   */
  async track(nextFrame: HTMLImageElement): Promise<HTMLCanvasElement> {
    if (!this.previousFrameGray || !this.currentMask) {
      throw new Error('Farneback tracker not initialized or reference not set')
    }

    try {
      console.log('[FarnebackMaskTracker] Computing optical flow and warping mask')

      // Convert next frame to grayscale
      const nextFrameMat = cv.imread(nextFrame)
      const nextFrameGray = new cv.Mat()
      cv.cvtColor(nextFrameMat, nextFrameGray, cv.COLOR_RGBA2GRAY)

      // Compute dense optical flow using Farneback algorithm
      const { flow, flowWidth, flowHeight } = this.computeOpticalFlow(
        this.previousFrameGray,
        nextFrameGray
      )

      // Warp mask using computed flow
      const warpedMask = this.warpMask(
        this.currentMask,
        flow,
        flowWidth,
        flowHeight
      )

      // Update state for next iteration
      this.previousFrameGray.delete()
      this.previousFrameGray = nextFrameGray
      this.previousFrame.delete()
      this.previousFrame = nextFrameMat
      this.currentMask = warpedMask

      // Convert to canvas
      return this.imageDataToCanvas(warpedMask)
    } catch (error) {
      console.error('[FarnebackMaskTracker] Error in track():', error)
      throw error
    }
  }

  /**
   * Compute optical flow using Farneback algorithm
   */
  private computeOpticalFlow(
    prevGray: any,
    nextGray: any
  ): { flow: Float32Array; flowWidth: number; flowHeight: number } {
    console.log('[FarnebackMaskTracker] Computing Farneback optical flow...')
    const startTime = performance.now()

    // Create flow output matrix (2-channel float)
    const flow = new cv.Mat()

    // Farneback optical flow parameters
    const pyrScale = 0.5 // Image scale for pyramid (< 1)
    const levels = 3 // Number of pyramid levels
    const winsize = 15 // Averaging window size
    const iterations = 3 // Iterations at each pyramid level
    const polyN = 5 // Size of pixel neighborhood (5 or 7)
    const polySigma = 1.2 // Gaussian std for poly_n (1.1 for 5, 1.5 for 7)
    const flags = 0 // Operation flags

    try {
      // Compute dense optical flow
      cv.calcOpticalFlowFarneback(
        prevGray,
        nextGray,
        flow,
        pyrScale,
        levels,
        winsize,
        iterations,
        polyN,
        polySigma,
        flags
      )

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2)
      console.log(`[FarnebackMaskTracker] Optical flow computed in ${elapsed}s`)

      // Extract flow data
      const flowWidth = flow.cols
      const flowHeight = flow.rows
      
      // Convert cv.Mat to Float32Array
      // Flow is 2-channel CV_32FC2 (dx, dy at each pixel)
      const flowData = new Float32Array(flowWidth * flowHeight * 2)
      
      // OpenCV stores flow as interleaved channels: [dx0, dy0, dx1, dy1, ...]
      // We need to convert to RAFT format: [all dx values, then all dy values]
      for (let y = 0; y < flowHeight; y++) {
        for (let x = 0; x < flowWidth; x++) {
          const srcIdx = (y * flowWidth + x) * 2
          const dstIdx = y * flowWidth + x
          
          // Extract from interleaved format
          const dx = flow.data32F[srcIdx]
          const dy = flow.data32F[srcIdx + 1]
          
          // Store in separated format: [all dx, all dy]
          flowData[dstIdx] = dx
          flowData[flowWidth * flowHeight + dstIdx] = dy
        }
      }

      flow.delete()

      console.log('[FarnebackMaskTracker] Flow field size:', {
        width: flowWidth,
        height: flowHeight,
        totalPixels: flowWidth * flowHeight,
      })

      return { flow: flowData, flowWidth, flowHeight }
    } catch (error) {
      flow.delete()
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
      `[FarnebackMaskTracker] Warping mask (${origWidth}x${origHeight}) using flow (${flowWidth}x${flowHeight})`
    )

    // Resize mask to match flow dimensions if they differ
    let processedMask = mask
    if (origWidth !== flowWidth || origHeight !== flowHeight) {
      processedMask = this.resizeImageData(mask, flowWidth, flowHeight)
      console.log(
        `[FarnebackMaskTracker] Resized mask to ${flowWidth}x${flowHeight} for warping`
      )
    }

    const warpedMask = new ImageData(flowWidth, flowHeight)

    // For each pixel in the new mask
    for (let y = 0; y < flowHeight; y++) {
      for (let x = 0; x < flowWidth; x++) {
        const flowIdx = y * flowWidth + x

        // Flow data is in [all dx, all dy] format (same as RAFT)
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
        `[FarnebackMaskTracker] Resized warped mask back to ${origWidth}x${origHeight}`
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
    console.log('[FarnebackMaskTracker] Cleaning up resources')
    
    if (this.previousFrame) {
      this.previousFrame.delete()
      this.previousFrame = null
    }
    
    if (this.previousFrameGray) {
      this.previousFrameGray.delete()
      this.previousFrameGray = null
    }
    
    this.currentMask = null
  }
}
