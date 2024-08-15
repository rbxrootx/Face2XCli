const ort = require('onnxruntime-node');
const ndarray = require('ndarray');
const ops = require('ndarray-ops');
const fs = require('fs').promises;

async function imageToNdarray(imagePath) {
  const buffer = await fs.readFile(imagePath);
  const sharp = require('sharp');
  const { data, info } = await sharp(buffer).raw().toBuffer({ resolveWithObject: true });
  return ndarray(new Uint8Array(data), [info.width, info.height, info.channels]);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function prepareImage(imageArray, model) {
  const width = imageArray.shape[0];
  const height = imageArray.shape[1];
  if (model === 'superRes') {
    const tensor = new ort.Tensor('uint8', imageArray.data, [width, height, 4]);
    return { input: tensor };
  } else if (model === 'tagger') {
    const newND = ndarray(new Uint8Array(width * height * 3), [1, 3, height, width]);
    ops.assign(newND.pick(0, null, null), imageArray.lo(0, 0, 0).hi(width, height, 3).transpose(2, 1, 0));
    const tensor = new ort.Tensor('uint8', newND.data, [1, 3, height, width]);
    return { input: tensor };
  } else {
    console.error('Invalid model type');
    throw new Error('Invalid model type');
  }
}

async function fetchModel(filepathOrUri, setProgress) {
  const data = await fs.readFile(filepathOrUri);
  setProgress(1);
  return data.buffer;
}

let superSession = null;

async function initializeONNX(setProgress) {
  ort.env.wasm.numThreads = 1; // Set to 1 for Node.js environment

  setProgress(0);
  const superBuf = await fetchModel('./models/superRes.onnx', setProgress);
  superSession = await ort.InferenceSession.create(superBuf);
  setProgress(1);

  await sleep(300);
}

async function runSuperRes(imageArray) {
  const feeds = prepareImage(imageArray, 'superRes');

  let sr;
  try {
    const output = await superSession.run(feeds);
    sr = output.output;
  } catch (e) {
    console.log('Failed to run super resolution');
    console.log(e);
  }
  return sr;
}

async function upscaleFrame(imageArray) {
  const CHUNK_SIZE = 1024;
  const PAD_SIZE = 32;

  const inImgW = imageArray.shape[0];
  const inImgH = imageArray.shape[1];
  const outImgW = inImgW * 2;
  const outImgH = inImgH * 2;
  const nChunksW = Math.ceil(inImgW / CHUNK_SIZE);
  const nChunksH = Math.ceil(inImgH / CHUNK_SIZE);
  const chunkW = Math.floor(inImgW / nChunksW);
  const chunkH = Math.floor(inImgH / nChunksH);

  const outArr = ndarray(new Uint8Array(outImgW * outImgH * 4), [outImgW, outImgH, 4]);
  for (let i = 0; i < nChunksH; i += 1) {
    for (let j = 0; j < nChunksW; j += 1) {
      const x = j * chunkW;
      const y = i * chunkH;

      const yStart = Math.max(0, y - PAD_SIZE);
      const inH = yStart + chunkH + PAD_SIZE * 2 > inImgH ? inImgH - yStart : chunkH + PAD_SIZE * 2;
      const outH = 2 * (Math.min(inImgH, y + chunkH) - y);
      const xStart = Math.max(0, x - PAD_SIZE);
      const inW = xStart + chunkW + PAD_SIZE * 2 > inImgW ? inImgW - xStart : chunkW + PAD_SIZE * 2;
      const outW = 2 * (Math.min(inImgW, x + chunkW) - x);

      const inSlice = imageArray.lo(xStart, yStart, 0).hi(inW, inH, 4);
      const subArr = ndarray(new Uint8Array(inW * inH * 4), inSlice.shape);
      ops.assign(subArr, inSlice);

      const chunkData = await runSuperRes(subArr);
      const chunkArr = ndarray(chunkData.data, chunkData.dims);
      const chunkSlice = chunkArr.lo((x - xStart) * 2, (y - yStart) * 2, 0).hi(outW, outH, 4);
      const outSlice = outArr.lo(x * 2, y * 2, 0).hi(outW, outH, 4);

      // Copy RGB channels
      ops.assign(outSlice.pick(null, null, [0, 1, 2]), chunkSlice.pick(null, null, [0, 1, 2]));

      // Process alpha channel separately
      processAlphaChannel(
        inSlice.lo((x - xStart), (y - yStart), 0).hi(outW / 2, outH / 2, 4),
        outSlice
      );
    }
  }

  return outArr;
}

function processAlphaChannel(inputArray, outputArray) {
  const inW = inputArray.shape[0];
  const inH = inputArray.shape[1];
  const outW = outputArray.shape[0];
  const outH = outputArray.shape[1];

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const inX = Math.floor(x / 2);
      const inY = Math.floor(y / 2);
      outputArray.set(x, y, 3, inputArray.get(inX, inY, 3));
    }
  }
}

async function multiUpscale(imageArray, upscaleFactor) {
  let outArr = imageArray;
  console.time('Upscaling');
  for (let s = 0; s < upscaleFactor; s += 1) {
    outArr = await upscaleFrame(outArr);
  }
  console.timeEnd('Upscaling');

  return outArr; // Return the raw ndarray
}

module.exports = {
  imageToNdarray,
  multiUpscale,
  initializeONNX,
  sleep,
  upscaleFrame,
  prepareImage,
  fetchModel
};