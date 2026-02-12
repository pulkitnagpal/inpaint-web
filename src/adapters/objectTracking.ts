import cv, { Mat } from 'opencv-ts'

/**
 * Extract bounding box from a binary mask
 */
export function getBoundingBoxFromMask(
  maskCanvas: HTMLCanvasElement
): { x: number; y: number; width: number; height: number } | null {
  console.log('[ObjectTracking] Extracting bounding box from mask...')

  const src = cv.imread(maskCanvas)
  const gray = new cv.Mat()
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)

  // Find contours
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(
    gray,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  )

  if (contours.size() === 0) {
    console.log('[ObjectTracking] No contours found in mask')
    src.delete()
    gray.delete()
    contours.delete()
    hierarchy.delete()
    return null
  }

  // Get the largest contour
  let maxArea = 0
  let maxContourIndex = 0
  for (let i = 0; i < contours.size(); i++) {
    const area = cv.contourArea(contours.get(i))
    if (area > maxArea) {
      maxArea = area
      maxContourIndex = i
    }
  }

  const rect = cv.boundingRect(contours.get(maxContourIndex))

  console.log('[ObjectTracking] Bounding box extracted:', rect)

  src.delete()
  gray.delete()
  contours.delete()
  hierarchy.delete()

  return rect
}

/**
 * Create a mask from a bounding box
 */
export function createMaskFromBBox(
  width: number,
  height: number,
  bbox: { x: number; y: number; width: number; height: number }
): HTMLCanvasElement {
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = width
  maskCanvas.height = height
  const ctx = maskCanvas.getContext('2d')!

  // Fill with black
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, width, height)

  // Draw white rectangle for the tracked object
  ctx.fillStyle = 'white'
  ctx.fillRect(bbox.x, bbox.y, bbox.width, bbox.height)

  return maskCanvas
}

/**
 * Track object using optical flow (Lucas-Kanade)
 * Simple feature tracking approach
 */
export class SimpleObjectTracker {
  private prevFrame: Mat | null = null
  private prevGray: Mat | null = null
  private prevPoints: Mat | null = null
  private bbox: { x: number; y: number; width: number; height: number } | null =
    null

  initialize(
    frame: HTMLImageElement,
    bbox: { x: number; y: number; width: number; height: number }
  ) {
    console.log('[SimpleObjectTracker] Initializing tracker with bbox:', bbox)

    this.bbox = { ...bbox }

    // Read frame
    this.prevFrame = cv.imread(frame)
    this.prevGray = new cv.Mat()
    cv.cvtColor(this.prevFrame, this.prevGray, cv.COLOR_RGBA2GRAY)

    // Generate feature points in the bounding box
    const mask = new cv.Mat.zeros(
      this.prevGray.rows,
      this.prevGray.cols,
      cv.CV_8UC1
    )
    cv.rectangle(
      mask,
      new cv.Point(bbox.x, bbox.y),
      new cv.Point(bbox.x + bbox.width, bbox.y + bbox.height),
      new cv.Scalar(255),
      -1
    )

    // Detect good features to track
    const allPoints = new cv.Mat()
    cv.goodFeaturesToTrack(
      this.prevGray,
      allPoints,
      200, // max corners (more to ensure enough in bbox)
      0.01, // quality level
      10 // min distance
    )

    // Filter points to only those within the bounding box
    const filteredPoints: number[] = []
    for (let i = 0; i < allPoints.rows; i++) {
      const x = allPoints.data32F[i * 2]
      const y = allPoints.data32F[i * 2 + 1]

      // Check if point is within bounding box
      if (
        x >= bbox.x &&
        x <= bbox.x + bbox.width &&
        y >= bbox.y &&
        y <= bbox.y + bbox.height
      ) {
        filteredPoints.push(x, y)
      }
    }

    // Create Mat from filtered points
    this.prevPoints = cv.matFromArray(
      filteredPoints.length / 2,
      1,
      cv.CV_32FC2,
      filteredPoints
    )

    console.log(
      '[SimpleObjectTracker] Detected',
      this.prevPoints.rows,
      'feature points in selected region (from',
      allPoints.rows,
      'total)'
    )

    allPoints.delete()
    mask.delete()
  }

  track(
    frame: HTMLImageElement
  ): { x: number; y: number; width: number; height: number } | null {
    if (!this.prevGray || !this.prevPoints || !this.bbox) {
      return null
    }

    // Read current frame
    const currFrame = cv.imread(frame)
    const currGray = new cv.Mat()
    cv.cvtColor(currFrame, currGray, cv.COLOR_RGBA2GRAY)

    // Calculate optical flow
    const nextPoints = new cv.Mat()
    const status = new cv.Mat()
    const err = new cv.Mat()

    cv.calcOpticalFlowPyrLK(
      this.prevGray,
      currGray,
      this.prevPoints,
      nextPoints,
      status,
      err,
      new cv.Size(15, 15),
      2,
      new cv.TermCriteria(
        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        10,
        0.03
      )
    )

    // Find valid tracked points
    const goodPrev: number[] = []
    const goodNext: number[] = []

    for (let i = 0; i < status.rows; i++) {
      if (status.data[i] === 1) {
        goodPrev.push(
          this.prevPoints.data32F[i * 2],
          this.prevPoints.data32F[i * 2 + 1]
        )
        goodNext.push(nextPoints.data32F[i * 2], nextPoints.data32F[i * 2 + 1])
      }
    }

    if (goodNext.length < 4) {
      console.log('[SimpleObjectTracker] Lost tracking - not enough points')
      currFrame.delete()
      currGray.delete()
      nextPoints.delete()
      status.delete()
      err.delete()
      return this.bbox // Return last known position
    }

    // Calculate movement delta (median of all point movements)
    const deltaX: number[] = []
    const deltaY: number[] = []
    for (let i = 0; i < goodNext.length / 2; i++) {
      deltaX.push(goodNext[i * 2] - goodPrev[i * 2])
      deltaY.push(goodNext[i * 2 + 1] - goodPrev[i * 2 + 1])
    }

    deltaX.sort((a, b) => a - b)
    deltaY.sort((a, b) => a - b)
    const medianDx = deltaX[Math.floor(deltaX.length / 2)]
    const medianDy = deltaY[Math.floor(deltaY.length / 2)]

    // Update bounding box position
    this.bbox.x = Math.max(
      0,
      Math.min(currGray.cols - this.bbox.width, this.bbox.x + medianDx)
    )
    this.bbox.y = Math.max(
      0,
      Math.min(currGray.rows - this.bbox.height, this.bbox.y + medianDy)
    )

    console.log('[SimpleObjectTracker] Updated bbox:', this.bbox, 'delta:', {
      x: medianDx,
      y: medianDy,
    })

    // Update for next iteration
    if (this.prevFrame) this.prevFrame.delete()
    if (this.prevGray) this.prevGray.delete()
    if (this.prevPoints) this.prevPoints.delete()

    this.prevFrame = currFrame
    this.prevGray = currGray
    this.prevPoints = nextPoints

    status.delete()
    err.delete()

    return { ...this.bbox }
  }

  cleanup() {
    if (this.prevFrame) {
      this.prevFrame.delete()
      this.prevFrame = null
    }
    if (this.prevGray) {
      this.prevGray.delete()
      this.prevGray = null
    }
    if (this.prevPoints) {
      this.prevPoints.delete()
      this.prevPoints = null
    }
  }
}
