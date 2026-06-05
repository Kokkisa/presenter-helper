'use strict';

// Generates assets/icon.png — 512x512 source icon for LiveCallAssistant.
// Pure Node, no native deps. Uses zlib + manual PNG chunk writing.
//
// Design: dark background (#0e0e16) with five rounded cyan (#00d4ff) bars
// forming an audio-waveform silhouette. Center bar tallest, mild asymmetry
// for visual interest (looks like a real voice spectrum).

const fs   = require('fs');
const path = require('path');
const zlib = require('zlib');

const W = 512;
const H = 512;

const BG = [0x0e, 0x0e, 0x16, 0xff];
const FG = [0x00, 0xd4, 0xff, 0xff];

const BARS = [
  { h: 150 },
  { h: 280 },
  { h: 380 },
  { h: 230 },
  { h: 170 }
];
const BAR_W   = 56;
const BAR_GAP = 24;
const RADIUS  = 22;

// ---- pixel buffer (RGBA) ----
const pixels = Buffer.alloc(W * H * 4);
for (let i = 0; i < W * H; i++) {
  pixels[i * 4 + 0] = BG[0];
  pixels[i * 4 + 1] = BG[1];
  pixels[i * 4 + 2] = BG[2];
  pixels[i * 4 + 3] = BG[3];
}

function setPx(x, y, c, alpha) {
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  const o = (y * W + x) * 4;
  if (alpha >= 1) {
    pixels[o]     = c[0];
    pixels[o + 1] = c[1];
    pixels[o + 2] = c[2];
    pixels[o + 3] = 0xff;
    return;
  }
  // alpha blend onto existing pixel
  const a = alpha;
  const ia = 1 - a;
  pixels[o]     = Math.round(c[0] * a + pixels[o]     * ia);
  pixels[o + 1] = Math.round(c[1] * a + pixels[o + 1] * ia);
  pixels[o + 2] = Math.round(c[2] * a + pixels[o + 2] * ia);
  pixels[o + 3] = 0xff;
}

// Filled rounded rect with antialiased corners.
function fillRoundedRect(x0, y0, w, h, r, c) {
  const x1 = x0 + w - 1;
  const y1 = y0 + h - 1;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      // distance from nearest corner center
      const dx = Math.min(x - x0, x1 - x);
      const dy = Math.min(y - y0, y1 - y);
      if (dx < r && dy < r) {
        const cdx = r - dx;
        const cdy = r - dy;
        const dist = Math.sqrt(cdx * cdx + cdy * cdy);
        if (dist <= r - 0.5) {
          setPx(x, y, c, 1);
        } else if (dist <= r + 0.5) {
          // edge — antialias
          const alpha = (r + 0.5 - dist);
          setPx(x, y, c, Math.max(0, Math.min(1, alpha)));
        }
        // else outside corner → leave background
      } else {
        setPx(x, y, c, 1);
      }
    }
  }
}

// ---- draw the five waveform bars ----
const totalW = BARS.length * BAR_W + (BARS.length - 1) * BAR_GAP;
const startX = Math.floor((W - totalW) / 2);
const cy     = H / 2;

for (let i = 0; i < BARS.length; i++) {
  const h = BARS[i].h;
  const x = startX + i * (BAR_W + BAR_GAP);
  const y = Math.floor(cy - h / 2);
  fillRoundedRect(x, y, BAR_W, h, RADIUS, FG);
}

// ---- PNG encoding ----
// Build raw scanlines: each row prefixed with a filter byte (0 = None).
const rowBytes = W * 4;
const raw = Buffer.alloc((rowBytes + 1) * H);
for (let y = 0; y < H; y++) {
  raw[y * (rowBytes + 1)] = 0;
  pixels.copy(raw, y * (rowBytes + 1) + 1, y * rowBytes, (y + 1) * rowBytes);
}
const compressed = zlib.deflateSync(raw, { level: 9 });

// CRC32 (PNG uses the standard polynomial 0xEDB88320, reflected).
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function makeChunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf  = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([length, typeBuf, data, crcBuf]);
}

const sig = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(W, 0);
ihdr.writeUInt32BE(H, 4);
ihdr[8]  = 8;   // bit depth
ihdr[9]  = 6;   // color type: RGBA
ihdr[10] = 0;   // compression: deflate
ihdr[11] = 0;   // filter: standard
ihdr[12] = 0;   // interlace: none

const png = Buffer.concat([
  sig,
  makeChunk('IHDR', ihdr),
  makeChunk('IDAT', compressed),
  makeChunk('IEND', Buffer.alloc(0))
]);

const outDir  = path.join(__dirname, '..', 'assets');
const outFile = path.join(outDir, 'icon.png');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outFile, png);

console.log(`Wrote ${outFile} (${png.length.toLocaleString()} bytes, ${W}x${H})`);
