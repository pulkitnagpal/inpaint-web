export async function loadVideo(videoFile: File): Promise<HTMLVideoElement> {
  console.log('[VideoProcessing] Loading video:', {
    name: videoFile.name,
    size: `${(videoFile.size / (1024 * 1024)).toFixed(2)} MB`,
    type: videoFile.type,
  })

  return new Promise((resolve, reject) => {
    const videoElement = document.createElement('video')
    videoElement.src = URL.createObjectURL(videoFile)
    videoElement.crossOrigin = 'anonymous'
    videoElement.onloadedmetadata = () => {
      console.log('[VideoProcessing] Video loaded:', {
        duration: `${videoElement.duration.toFixed(2)}s`,
        dimensions: `${videoElement.videoWidth}x${videoElement.videoHeight}`,
        fps: 'auto-detected',
      })
      resolve(videoElement)
    }
    videoElement.onerror = error => {
      console.error('[VideoProcessing] Error loading video:', error)
      reject(error)
    }
  })
}

export async function getFirstFrame(
  videoFile: File
): Promise<HTMLImageElement> {
  console.log('[VideoProcessing] Extracting first frame...')
  const video = await loadVideo(videoFile)
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!

  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  console.log('[VideoProcessing] Canvas setup:', {
    width: canvas.width,
    height: canvas.height,
  })

  video.currentTime = 0

  return new Promise(resolve => {
    video.onseeked = () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const img = new Image()
      img.onload = () => {
        console.log('[VideoProcessing] First frame extracted successfully')
        resolve(img)
      }
      img.src = canvas.toDataURL()
    }
  })
}

export async function extractFrames(
  video: HTMLVideoElement,
  fps: number = 30,
  onProgress?: (progress: number) => void
): Promise<HTMLImageElement[]> {
  const estimatedFrames = Math.ceil(video.duration * fps)
  console.log('[VideoProcessing] Starting frame extraction:', {
    duration: `${video.duration.toFixed(2)}s`,
    fps,
    estimatedFrames,
    dimensions: `${video.videoWidth}x${video.videoHeight}`,
  })

  const startTime = performance.now()

  return new Promise((resolve, reject) => {
    const frames: HTMLImageElement[] = []
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const duration = video.duration
    const frameInterval = 1 / fps
    let currentTime = 0

    const captureFrame = () => {
      if (currentTime > duration) {
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2)
        console.log('[VideoProcessing] Frame extraction complete:', {
          totalFrames: frames.length,
          timeElapsed: `${elapsed}s`,
          avgTimePerFrame: `${(
            (parseFloat(elapsed) / frames.length) *
            1000
          ).toFixed(0)}ms`,
        })
        resolve(frames)
        return
      }

      video.currentTime = currentTime
    }

    video.onseeked = () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const img = new Image()
      img.onload = () => {
        frames.push(img)
        const progress = (currentTime / duration) * 50
        if (frames.length % 10 === 0) {
          console.log(
            `[VideoProcessing] Extracted ${
              frames.length
            }/${estimatedFrames} frames (${progress.toFixed(1)}%)`
          )
        }
        if (onProgress) {
          onProgress(progress) // 50% for extraction
        }
        currentTime += frameInterval
        captureFrame()
      }
      img.src = canvas.toDataURL()
    }

    video.onerror = error => {
      console.error('[VideoProcessing] Error during frame extraction:', error)
      reject(error)
    }
    captureFrame()
  })
}

export async function processFrames(
  frames: HTMLImageElement[],
  processFrame: (frame: HTMLImageElement, index: number) => Promise<string>,
  onProgress?: (progress: number) => void
): Promise<string[]> {
  console.log('[VideoProcessing] Starting inpainting on frames:', {
    totalFrames: frames.length,
  })

  const processedFrames: string[] = []
  const startTime = performance.now()

  for (let i = 0; i < frames.length; i++) {
    const frameStartTime = performance.now()
    const processedDataUrl = await processFrame(frames[i], i)
    const frameTime = performance.now() - frameStartTime

    processedFrames.push(processedDataUrl)

    const progress = 50 + ((i + 1) / frames.length) * 50
    const eta =
      (((performance.now() - startTime) / (i + 1)) * (frames.length - i - 1)) /
      1000

    console.log(
      `[VideoProcessing] Processed frame ${i + 1}/${frames.length}:`,
      {
        progress: `${progress.toFixed(1)}%`,
        frameTime: `${frameTime.toFixed(0)}ms`,
        eta: `${eta.toFixed(1)}s`,
      }
    )

    if (onProgress) {
      onProgress(progress) // 50-100% for processing
    }
  }

  const totalTime = ((performance.now() - startTime) / 1000).toFixed(2)
  console.log('[VideoProcessing] Inpainting complete:', {
    totalFrames: frames.length,
    totalTime: `${totalTime}s`,
    avgTimePerFrame: `${(parseFloat(totalTime) / frames.length).toFixed(2)}s`,
  })

  return processedFrames
}

export async function combineFramesToVideo(
  frameDataUrls: string[],
  fps: number,
  onProgress?: (blob: Blob) => void
): Promise<Blob> {
  console.log('[VideoProcessing] Combining frames to video:', {
    totalFrames: frameDataUrls.length,
    fps,
    estimatedDuration: `${(frameDataUrls.length / fps).toFixed(2)}s`,
  })

  return new Promise((resolve, reject) => {
    if (frameDataUrls.length === 0) {
      console.error('[VideoProcessing] No frames to combine')
      reject(new Error('No frames to combine'))
      return
    }

    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!

    const firstImg = new Image()
    firstImg.onload = () => {
      canvas.width = firstImg.width
      canvas.height = firstImg.height

      const stream = canvas.captureStream(fps)
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm'

      console.log('[VideoProcessing] Starting video recording:', {
        dimensions: `${canvas.width}x${canvas.height}`,
        codec: mimeType,
        bitrate: '5 Mbps',
      })

      const recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 5000000, // 5 Mbps
      })

      const chunks: Blob[] = []

      recorder.ondataavailable = event => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType })
        console.log('[VideoProcessing] Video recording complete:', {
          size: `${(blob.size / (1024 * 1024)).toFixed(2)} MB`,
          type: blob.type,
        })
        resolve(blob)
      }

      recorder.onerror = error => {
        console.error('[VideoProcessing] Recording error:', error)
        reject(error)
      }
      recorder.start()

      let frameIndex = 0
      const frameInterval = 1000 / fps
      const startTime = performance.now()

      const drawNextFrame = () => {
        if (frameIndex >= frameDataUrls.length) {
          const elapsed = ((performance.now() - startTime) / 1000).toFixed(2)
          console.log('[VideoProcessing] All frames drawn:', {
            totalFrames: frameIndex,
            timeElapsed: `${elapsed}s`,
          })
          recorder.stop()
          return
        }

        if (frameIndex % 30 === 0) {
          console.log(
            `[VideoProcessing] Drawing frame ${frameIndex}/${frameDataUrls.length}`
          )
        }

        const img = new Image()
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
          frameIndex++
          setTimeout(drawNextFrame, frameInterval)
        }
        img.src = frameDataUrls[frameIndex]
      }

      drawNextFrame()
    }

    firstImg.src = frameDataUrls[0]
  })
}
