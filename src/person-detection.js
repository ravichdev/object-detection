import * as tf from "@tensorflow/tfjs";
import {
  SupportedModels,
  createSegmenter,
  toColoredMask,
  bodyPixMaskValueToRainbowColor,
  drawMask,
  drawPixelatedMask,
} from "@tensorflow-models/body-segmentation";
import * as mpSelfieSegmentation from "@mediapipe/selfie_segmentation";
import { setupCameraApp } from "./camera";

const video = document.querySelector("#video");
const canvas = document.querySelector("#output");
const canvasCtx = canvas.getContext("2d");
let rafId;
const activeEffect = "";

const CANVAS_NAMES = {
  blurred: "blurred",
  blurredMask: "blurred-mask",
  mask: "mask",
  lowresPartMask: "lowres-part-mask",
};
const offScreenCanvases = {};

function createOffScreenCanvas() {
  if (typeof document !== "undefined") {
    return document.createElement("canvas");
  } else if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(0, 0);
  } else {
    throw new Error("Cannot create a canvas in this context");
  }
}

function ensureOffscreenCanvasCreated(id) {
  if (!offScreenCanvases[id]) {
    offScreenCanvases[id] = createOffScreenCanvas();
  }
  return offScreenCanvases[id];
}

function drawAndBlurImageOnCanvas(image, blurAmount, canvas) {
  const { height, width } = image;
  const ctx = canvas.getContext("2d");
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.filter = `blur(${blurAmount}px)`;
  ctx.drawImage(image, 0, 0, width, height);
  ctx.restore();
}

function drawAndBlurImageOnOffScreenCanvas(
  image,
  blurAmount,
  offscreenCanvasName
) {
  const canvas = ensureOffscreenCanvasCreated(offscreenCanvasName);
  if (blurAmount === 0) {
    renderImageToCanvas(image, canvas);
  } else {
    drawAndBlurImageOnCanvas(image, blurAmount, canvas);
  }
  return canvas;
}

function renderImageToCanvas(image, canvas) {
  const { width, height } = image;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  ctx.drawImage(image, 0, 0, width, height);
}

function flipCanvasHorizontal(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
}

function drawWithCompositing(ctx, image, compositeOperation) {
  ctx.globalCompositeOperation = compositeOperation;
  ctx.drawImage(image, 0, 0);
}

function createPersonMask(multiPersonSegmentation, edgeBlurAmount) {
  const backgroundMaskImage = toMask(
    multiPersonSegmentation,
    { r: 0, g: 0, b: 0, a: 255 },
    { r: 0, g: 0, b: 0, a: 0 }
  );

  const backgroundMask = renderImageDataToOffScreenCanvas(
    backgroundMaskImage,
    CANVAS_NAMES.mask
  );
  if (edgeBlurAmount === 0) {
    return backgroundMask;
  } else {
    return drawAndBlurImageOnOffScreenCanvas(
      backgroundMask,
      edgeBlurAmount,
      CANVAS_NAMES.blurredMask
    );
  }
}

function toMask(
  personOrPartSegmentation,
  foreground = {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
  },
  background = {
    r: 0,
    g: 0,
    b: 0,
    a: 255,
  },
  drawContour = false,
  foregroundIds = [1]
) {
  if (
    Array.isArray(personOrPartSegmentation) &&
    personOrPartSegmentation.length === 0
  ) {
    return null;
  }

  let multiPersonOrPartSegmentation;

  if (!Array.isArray(personOrPartSegmentation)) {
    multiPersonOrPartSegmentation = [personOrPartSegmentation];
  } else {
    multiPersonOrPartSegmentation = personOrPartSegmentation;
  }

  const { width, height } = multiPersonOrPartSegmentation[0];
  const bytes = new Uint8ClampedArray(width * height * 4);

  function drawStroke(
    bytes,
    row,
    column,
    width,
    radius,
    color = { r: 0, g: 255, b: 255, a: 255 }
  ) {
    for (let i = -radius; i <= radius; i++) {
      for (let j = -radius; j <= radius; j++) {
        if (i !== 0 && j !== 0) {
          const n = (row + i) * width + (column + j);
          bytes[4 * n + 0] = color.r;
          bytes[4 * n + 1] = color.g;
          bytes[4 * n + 2] = color.b;
          bytes[4 * n + 3] = color.a;
        }
      }
    }
  }

  function isSegmentationBoundary(
    segmentationData,
    row,
    column,
    width,
    foregroundIds = [1],
    radius = 1
  ) {
    let numberBackgroundPixels = 0;
    for (let i = -radius; i <= radius; i++) {
      for (let j = -radius; j <= radius; j++) {
        if (i !== 0 && j !== 0) {
          const n = (row + i) * width + (column + j);
          if (!foregroundIds.some((id) => id === segmentationData[n])) {
            numberBackgroundPixels += 1;
          }
        }
      }
    }
    return numberBackgroundPixels > 0;
  }

  for (let i = 0; i < height; i += 1) {
    for (let j = 0; j < width; j += 1) {
      const n = i * width + j;
      bytes[4 * n + 0] = background.r;
      bytes[4 * n + 1] = background.g;
      bytes[4 * n + 2] = background.b;
      bytes[4 * n + 3] = background.a;
      for (let k = 0; k < multiPersonOrPartSegmentation.length; k++) {
        if (
          foregroundIds.some(
            (id) => id === multiPersonOrPartSegmentation[k].data[n]
          )
        ) {
          bytes[4 * n] = foreground.r;
          bytes[4 * n + 1] = foreground.g;
          bytes[4 * n + 2] = foreground.b;
          bytes[4 * n + 3] = foreground.a;
          const isBoundary = isSegmentationBoundary(
            multiPersonOrPartSegmentation[k].data,
            i,
            j,
            width,
            foregroundIds
          );
          if (
            drawContour &&
            i - 1 >= 0 &&
            i + 1 < height &&
            j - 1 >= 0 &&
            j + 1 < width &&
            isBoundary
          ) {
            drawStroke(bytes, i, j, width, 1);
          }
        }
      }
    }
  }

  return new ImageData(bytes, width, height);
}

function drawBokehEffect(
  canvas,
  image,
  multiPersonSegmentation,
  backgroundBlurAmount = 3,
  edgeBlurAmount = 3,
  flipHorizontal = false
) {
  const blurredImage = drawAndBlurImageOnOffScreenCanvas(
    image,
    backgroundBlurAmount,
    CANVAS_NAMES.blurred
  );
  canvas.width = blurredImage.width;
  canvas.height = blurredImage.height;

  const ctx = canvas.getContext("2d");

  if (
    Array.isArray(multiPersonSegmentation) &&
    multiPersonSegmentation.length === 0
  ) {
    ctx.drawImage(blurredImage, 0, 0);
    return;
  }

  const personMask = createPersonMask(multiPersonSegmentation, edgeBlurAmount);

  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  // draw the original image on the final canvas
  const [height, width] = getInputSize(image);
  ctx.drawImage(image, 0, 0, width, height);

  // "destination-in" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // crop what's not the person using the mask from the original image
  drawWithCompositing(ctx, personMask, "destination-in");
  // "destination-over" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // draw the blurred background on top of the original image where it doesn't
  // overlap.
  drawWithCompositing(ctx, blurredImage, "destination-over");
  ctx.restore();
}

async function app() {
  setupCameraApp();

  runBodySegmentation();
}

function onResults(results) {
  // Get Video Properties
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  // Set video width
  video.width = videoWidth;
  video.height = videoHeight;

  // Set canvas height and width
  canvas.width = videoWidth;
  canvas.height = videoHeight;

  // Draw the overlays.
  canvasCtx.save();

  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  canvasCtx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  drawWithCompositing(canvasCtx, results.segmentationMask, "destination-in");

  const blurredImage = drawAndBlurImageOnOffScreenCanvas(
    results.image,
    20,
    CANVAS_NAMES.blurred
  );

  drawWithCompositing(canvasCtx, blurredImage, "destination-over");

  canvasCtx.restore();
}

const runBodySegmentation = async () => {
  // const model = SupportedModels.MediaPipeSelfieSegmentation; // or 'BodyPix'

  // const segmenterConfig = {
  //     runtime: 'mediapipe', // or 'tfjs'
  //     modelType: 'general', // or 'landscape'
  //     solutionPath:
  //     `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@${
  //         mpSelfieSegmentation.VERSION}`
  // };

  // const segmenter = await createSegmenter(model, segmenterConfig);

  // const video = document.getElementById('video');
  // const segmentation = await segmenter.segmentPeople(video);
  // console.log(segmentation);

  // const data = await toColoredMask(
  //     segmentation, bodyPixMaskValueToRainbowColor,
  //     {r: 0, g: 0, b: 0, a: 255});
  // await drawMask(
  //     canvas, video, data, 0.5, 0.5);

  const selfieSegmentation = new mpSelfieSegmentation.SelfieSegmentation({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
    },
  });
  selfieSegmentation.setOptions({
    modelSelection: 1,
  });
  selfieSegmentation.onResults(onResults);

  const renderPrediction = async () => {
    await selfieSegmentation.send({ image: video });

    rafId = requestAnimationFrame(renderPrediction);
  };

  renderPrediction();
};

app();
