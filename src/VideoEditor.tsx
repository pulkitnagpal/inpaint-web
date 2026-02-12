/* eslint-disable jsx-a11y/click-events-have-key-events */
/* eslint-disable jsx-a11y/no-static-element-interactions */
import { DownloadIcon, PlayIcon } from '@heroicons/react/outline'
import { useCallback, useEffect, useState, useRef, useMemo } from 'react'
import { useWindowSize } from 'react-use'
import inpaint from './adapters/inpainting'
import Button from './components/Button'
import Slider from './components/Slider'
import Progress from './components/Progress'
import Modal from './components/Modal'
import {
  getFirstFrame,
  extractFrames,
  processFrames,
  combineFramesToVideo,
} from './adapters/videoProcessing'
import {
  getBoundingBoxFromMask,
  createMaskFromBBox,
  SimpleObjectTracker,
} from './adapters/objectTracking'
import { FarnebackMaskTracker } from './adapters/farnebackTracking'

interface VideoEditorProps {
  videoFile: File
}

interface Line {
  size?: number
  pts: { x: number; y: number }[]
}

function drawLines(
  ctx: CanvasRenderingContext2D,
  lines: Line[],
  color = 'rgba(255, 0, 0, 0.5)'
) {
  ctx.strokeStyle = color
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  lines.forEach(line => {
    if (!line?.pts.length || !line.size) {
      return
    }
    ctx.lineWidth = line.size
    ctx.beginPath()
    ctx.moveTo(line.pts[0].x, line.pts[0].y)
    line.pts.forEach(pt => ctx.lineTo(pt.x, pt.y))
    ctx.stroke()
  })
}

export default function VideoEditor(props: VideoEditorProps) {
  const { videoFile } = props
  const [brushSize, setBrushSize] = useState(40)
  const [firstFrame, setFirstFrame] = useState<HTMLImageElement>()
  const [context, setContext] = useState<CanvasRenderingContext2D>()
  const [maskCanvas] = useState<HTMLCanvasElement>(() => {
    return document.createElement('canvas')
  })
  const [lines, setLines] = useState<Line[]>([{ pts: [] }])
  const brushRef = useRef<HTMLDivElement>(null)
  const [showBrush, setShowBrush] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [processedVideoBlob, setProcessedVideoBlob] = useState<Blob>()
  const canvasDiv = useRef<HTMLDivElement>(null)
  const windowSize = useWindowSize()
  const [isDrawing, setIsDrawing] = useState(false)

  // Memoize video URLs to prevent recreating them on every render
  const originalVideoUrl = useMemo(() => {
    return URL.createObjectURL(videoFile)
  }, [videoFile])

  const processedVideoUrl = useMemo(() => {
    return processedVideoBlob ? URL.createObjectURL(processedVideoBlob) : null
  }, [processedVideoBlob])

  // Cleanup object URLs when component unmounts or URLs change
  useEffect(() => {
    return () => {
      if (originalVideoUrl) {
        URL.revokeObjectURL(originalVideoUrl)
      }
    }
  }, [originalVideoUrl])

  useEffect(() => {
    return () => {
      if (processedVideoUrl) {
        URL.revokeObjectURL(processedVideoUrl)
      }
    }
  }, [processedVideoUrl])

  useEffect(() => {
    async function loadFirstFrame() {
      const frame = await getFirstFrame(videoFile)
      setFirstFrame(frame)
    }
    loadFirstFrame()
  }, [videoFile])

  const draw = useCallback(() => {
    if (!context || !firstFrame) {
      return
    }
    context.clearRect(0, 0, context.canvas.width, context.canvas.height)
    const { canvas } = context

    const divWidth = canvasDiv.current?.offsetWidth || 0
    const divHeight = canvasDiv.current?.offsetHeight || 0

    const imgAspectRatio = firstFrame.width / firstFrame.height
    const divAspectRatio = divWidth / divHeight

    let canvasWidth
    let canvasHeight

    if (divAspectRatio > imgAspectRatio) {
      canvasHeight = divHeight
      canvasWidth = divHeight * imgAspectRatio
    } else {
      canvasWidth = divWidth
      canvasHeight = divWidth / imgAspectRatio
    }

    canvas.width = canvasWidth
    canvas.height = canvasHeight

    context.drawImage(firstFrame, 0, 0, canvasWidth, canvasHeight)

    const currentLines = lines.map(line => ({
      ...line,
      pts: line.pts.map(pt => ({
        x: (pt.x * canvasWidth) / firstFrame.width,
        y: (pt.y * canvasHeight) / firstFrame.height,
      })),
      size: (line.size || brushSize) * (canvasWidth / firstFrame.width),
    }))

    drawLines(context, currentLines)
  }, [context, firstFrame, lines, brushSize])

  useEffect(() => {
    draw()
  }, [draw, windowSize])

  const refreshCanvasMask = useCallback(() => {
    if (!context?.canvas.width || !context.canvas.height || !firstFrame) {
      throw new Error('canvas has invalid size')
    }

    maskCanvas.width = firstFrame.width
    maskCanvas.height = firstFrame.height
    const ctx = maskCanvas.getContext('2d')
    if (!ctx) {
      throw new Error('could not retrieve mask canvas')
    }

    drawLines(ctx, lines, 'white')
  }, [
    context?.canvas.height,
    context?.canvas.width,
    maskCanvas,
    lines,
    firstFrame,
  ])

  const handleMouseDown = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (!context?.canvas) return
      setIsDrawing(true)

      const canvas = context.canvas
      const rect = canvas.getBoundingClientRect()
      const x =
        ((ev.clientX - rect.left) / canvas.width) * (firstFrame?.width || 0)
      const y =
        ((ev.clientY - rect.top) / canvas.height) * (firstFrame?.height || 0)

      setLines([...lines, { pts: [{ x, y }], size: brushSize }])
    },
    [context, lines, brushSize, firstFrame]
  )

  const handleMouseMove = useCallback(
    (ev: React.MouseEvent<HTMLCanvasElement>) => {
      if (!context?.canvas || !firstFrame) return

      const canvas = context.canvas
      const rect = canvas.getBoundingClientRect()
      const x = ((ev.clientX - rect.left) / canvas.width) * firstFrame.width
      const y = ((ev.clientY - rect.top) / canvas.height) * firstFrame.height

      // Update brush position relative to the canvas container
      if (brushRef.current && canvasDiv.current) {
        const containerRect = canvasDiv.current.getBoundingClientRect()
        const relativeX = ev.clientX - containerRect.left
        const relativeY = ev.clientY - containerRect.top

        brushRef.current.style.left = `${relativeX}px`
        brushRef.current.style.top = `${relativeY}px`
        brushRef.current.style.display = 'block'
      }

      if (!isDrawing) return

      setLines(currentLines => {
        const newLines = [...currentLines]
        const lastLine = newLines[newLines.length - 1]
        lastLine.pts.push({ x, y })
        return newLines
      })
    },
    [context, isDrawing, firstFrame]
  )

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false)
  }, [])

  const processVideoWithOpenCV = useCallback(async () => {
    if (!firstFrame) return

    console.log('[VideoEditor] Starting OpenCV-based video processing...')
    const workflowStartTime = performance.now()

    setIsProcessing(true)
    setProgress(0)

    try {
      // Get mask and extract bounding box
      console.log('[VideoEditor] Step 1: Generating mask from drawn lines')
      refreshCanvasMask()
      const maskDataUrl = maskCanvas.toDataURL()
      console.log(
        '[VideoEditor] Mask generated, size:',
        maskDataUrl.length,
        'chars'
      )

      // Extract bounding box from mask
      console.log('[VideoEditor] Step 1a: Extracting bounding box from mask')
      const bbox = getBoundingBoxFromMask(maskCanvas)
      if (!bbox) {
        throw new Error('Could not extract bounding box from mask')
      }
      console.log('[VideoEditor] Bounding box:', bbox)

      // Load video and extract frames
      console.log('[VideoEditor] Step 2: Loading video metadata')
      const video = document.createElement('video')
      video.src = URL.createObjectURL(videoFile)
      await new Promise(resolve => {
        video.onloadedmetadata = resolve
      })

      const fps = 30
      console.log('[VideoEditor] Step 3: Extracting frames at', fps, 'fps')
      const frames = await extractFrames(video, fps, setProgress)
      console.log('[VideoEditor] Extracted', frames.length, 'frames')

      // Initialize OpenCV object tracker
      console.log('[VideoEditor] Step 4: Initializing OpenCV tracker')
      const tracker = new SimpleObjectTracker()
      tracker.initialize(frames[0], bbox)

      // Process each frame with tracking and inpainting
      console.log(
        '[VideoEditor] Step 5: Processing frames with OpenCV tracking and inpainting'
      )
      const processedFrameUrls = await processFrames(
        frames,
        async (frame, index) => {
          // Track object in current frame
          let currentBBox = bbox
          if (index > 0) {
            const trackedBBox = tracker.track(frame)
            if (trackedBBox) {
              currentBBox = trackedBBox
            }
          }

          // Generate mask for current frame based on tracked position
          const frameMaskCanvas = createMaskFromBBox(
            frame.width,
            frame.height,
            currentBBox
          )
          const frameMaskDataUrl = frameMaskCanvas.toDataURL()

          // Apply inpainting with tracked mask
          return inpaint(frame, frameMaskDataUrl)
        },
        setProgress
      )

      // Cleanup tracker
      tracker.cleanup()

      console.log(
        '[VideoEditor] Processed',
        processedFrameUrls.length,
        'frames'
      )

      // Combine frames back into video
      console.log('[VideoEditor] Step 6: Combining frames to video')
      const videoBlob = await combineFramesToVideo(processedFrameUrls, fps)
      setProcessedVideoBlob(videoBlob)

      const totalTime = (
        (performance.now() - workflowStartTime) /
        1000
      ).toFixed(2)
      console.log('[VideoEditor] ✅ OpenCV video processing complete!', {
        totalTime: `${totalTime}s`,
        outputSize: `${(videoBlob.size / (1024 * 1024)).toFixed(2)} MB`,
      })
    } catch (error) {
      const elapsed = ((performance.now() - workflowStartTime) / 1000).toFixed(
        2
      )
      // eslint-disable-next-line no-console
      console.error(
        '[VideoEditor] ❌ Error processing video after',
        elapsed,
        's:',
        error
      )
      // eslint-disable-next-line no-alert
      alert('Error processing video. Please try again.')
    } finally {
      setIsProcessing(false)
      setProgress(0)
    }
  }, [firstFrame, videoFile, maskCanvas, refreshCanvasMask])

  const processVideoWithFarneback = useCallback(async () => {
    if (!firstFrame) return

    console.log('[VideoEditor] Starting Farneback optical flow video processing...')
    const workflowStartTime = performance.now()

    setIsProcessing(true)
    setProgress(0)

    try {
      // Get mask
      console.log('[VideoEditor] Step 1: Generating mask from drawn lines')
      refreshCanvasMask()
      console.log('[VideoEditor] Mask generated')

      // Load video and extract frames
      console.log('[VideoEditor] Step 2: Loading video metadata')
      const video = document.createElement('video')
      video.src = URL.createObjectURL(videoFile)
      await new Promise(resolve => {
        video.onloadedmetadata = resolve
      })

      const fps = 30
      console.log('[VideoEditor] Step 3: Extracting frames at', fps, 'fps')
      const frames = await extractFrames(video, fps, setProgress)
      console.log('[VideoEditor] Extracted', frames.length, 'frames')

      // Initialize Farneback tracker
      console.log('[VideoEditor] Step 4: Initializing Farneback optical flow tracker')
      const tracker = new FarnebackMaskTracker()
      await tracker.initialize()
      tracker.setReference(frames[0], maskCanvas)

      // Process each frame with Farneback tracking and inpainting
      console.log(
        '[VideoEditor] Step 5: Processing frames with Farneback optical flow and inpainting'
      )
      const processedFrameUrls = await processFrames(
        frames,
        async (frame, index) => {
          let frameMaskCanvas: HTMLCanvasElement

          if (index === 0) {
            // Use original mask for first frame
            frameMaskCanvas = maskCanvas
          } else {
            // Track mask using Farneback optical flow
            frameMaskCanvas = await tracker.track(frame)
          }

          const frameMaskDataUrl = frameMaskCanvas.toDataURL()

          // Apply inpainting with tracked mask
          return inpaint(frame, frameMaskDataUrl)
        },
        setProgress
      )

      // Cleanup tracker
      tracker.cleanup()

      console.log(
        '[VideoEditor] Processed',
        processedFrameUrls.length,
        'frames'
      )

      // Combine frames back into video
      console.log('[VideoEditor] Step 6: Combining frames to video')
      const videoBlob = await combineFramesToVideo(processedFrameUrls, fps)
      setProcessedVideoBlob(videoBlob)

      const totalTime = (
        (performance.now() - workflowStartTime) /
        1000
      ).toFixed(2)
      console.log('[VideoEditor] ✅ Farneback optical flow video processing complete!', {
        totalTime: `${totalTime}s`,
        outputSize: `${(videoBlob.size / (1024 * 1024)).toFixed(2)} MB`,
      })
    } catch (error) {
      const elapsed = ((performance.now() - workflowStartTime) / 1000).toFixed(
        2
      )
      // eslint-disable-next-line no-console
      console.error(
        '[VideoEditor] ❌ Error processing video after',
        elapsed,
        's:',
        error
      )
      // eslint-disable-next-line no-alert
      alert('Error processing video with Farneback. Please try again.')
    } finally {
      setIsProcessing(false)
      setProgress(0)
    }
  }, [firstFrame, videoFile, maskCanvas, refreshCanvasMask])

  const downloadVideo = useCallback(() => {
    if (!processedVideoBlob) return

    const url = URL.createObjectURL(processedVideoBlob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'inpainted-video.webm'
    a.click()
    URL.revokeObjectURL(url)
  }, [processedVideoBlob])

  const undoStroke = useCallback(() => {
    if (lines.length > 0) {
      setLines(lines.slice(0, -1))
    }
  }, [lines])

  if (!firstFrame) {
    return (
      <div className="flex items-center justify-center h-full">
        <p>Loading video...</p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-4">
          <span className="text-sm">Brush Size: {brushSize}</span>
          <Slider
            label=""
            value={brushSize}
            onChange={setBrushSize}
            min={10}
            max={150}
          />
        </div>
        <div className="flex space-x-2">
          <Button onClick={undoStroke} disabled={lines.length === 0}>
            Undo
          </Button>
          <Button
            icon={<PlayIcon className="w-5 h-5" />}
            onClick={processVideoWithOpenCV}
            disabled={isProcessing || lines.length === 0}
          >
            Process Video (OpenCV)
          </Button>
          <Button
            icon={<PlayIcon className="w-5 h-5" />}
            onClick={processVideoWithFarneback}
            disabled={isProcessing || lines.length === 0}
          >
            Process Video (Farneback)
          </Button>
          {processedVideoBlob && (
            <Button
              icon={<DownloadIcon className="w-5 h-5" />}
              onClick={downloadVideo}
            >
              Download
            </Button>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Original Video */}
        <div className="w-1/3 border-r p-4 flex flex-col">
          <h3 className="text-lg font-semibold mb-2">Original Video</h3>
          <div className="flex-1 flex items-center justify-center bg-gray-100 rounded">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              controls
              className="max-w-full max-h-full"
              src={originalVideoUrl}
            />
          </div>
        </div>

        {/* Middle: First Frame with Drawing Canvas */}
        <div className="w-1/3 border-r p-4 flex flex-col">
          <h3 className="text-lg font-semibold mb-2">
            Draw Mask on First Frame
          </h3>
          <div
            ref={canvasDiv}
            className="flex-1 flex items-center justify-center bg-gray-100 rounded relative overflow-hidden"
            onMouseEnter={() => {
              setShowBrush(true)
              if (brushRef.current) {
                brushRef.current.style.display = 'block'
              }
            }}
            onMouseLeave={() => {
              setShowBrush(false)
              if (brushRef.current) {
                brushRef.current.style.display = 'none'
              }
            }}
          >
            <canvas
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              ref={r => {
                if (r && !context) {
                  const ctx = r.getContext('2d')
                  if (ctx) {
                    setContext(ctx)
                  }
                }
              }}
              style={{ cursor: 'none' }}
            />

            {showBrush && !isProcessing && (
              <div
                ref={brushRef}
                className="pointer-events-none absolute rounded-full border-2 border-white"
                style={{
                  width: `${brushSize}px`,
                  height: `${brushSize}px`,
                  transform: 'translate(-50%, -50%)',
                  boxShadow: '0 0 0 1px black',
                  display: 'none',
                  zIndex: 10,
                }}
              />
            )}
          </div>
        </div>

        {/* Right: Processed Video */}
        <div className="w-1/3 p-4 flex flex-col">
          <h3 className="text-lg font-semibold mb-2">Processed Video</h3>
          <div className="flex-1 flex items-center justify-center bg-gray-100 rounded">
            {processedVideoUrl ? (
              <div className="max-w-full max-h-full w-full h-full flex items-center justify-center">
                {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                <video
                  controls
                  className="max-w-full max-h-full"
                  src={processedVideoUrl}
                />
              </div>
            ) : (
              <div className="text-gray-400 text-center">
                <p>Processed video will appear here</p>
                <p className="text-sm mt-2">
                  Draw a mask and click &quot;Process Video&quot;
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {isProcessing && (
        <Modal>
          <div className="text-xl space-y-5">
            <p>Processing video...</p>
            <Progress percent={progress} />
          </div>
        </Modal>
      )}
    </div>
  )
}
