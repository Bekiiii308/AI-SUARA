// ======================================================
// CONFIG
// ======================================================

const CONFIG = {

    modelPath: "best.onnx",

    labels: [
        "man",
        "woman"
    ],

    threshold: 0.75,

    iouThreshold: 0.4
};

// ======================================================
// ELEMENT HTML
// ======================================================

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctxOverlay = overlay.getContext("2d");

const processor = document.getElementById("processor");
const ctxProcessor = processor.getContext("2d", {
    willReadFrequently: true
});

const statusText = document.getElementById("status");
const initBtn = document.getElementById("btn-init");

let session;

const TARGET_SIZE = 640;

// ======================================================
// LOAD MODEL
// ======================================================

initBtn.addEventListener("click", async () => {

    initBtn.disabled = true;
    initBtn.innerText = "MEMUAT MODEL...";

    try {

        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        session = await ort.InferenceSession.create(
            CONFIG.modelPath,
            {
                executionProviders: ["webgl", "wasm"]
            }
        );

        statusText.innerText = "MODEL BERHASIL DIMUAT";

        startCamera();

    } catch (err) {

        console.error(err);

        statusText.innerText =
            "GAGAL MEMUAT MODEL";
    }
});

// ======================================================
// START CAMERA
// ======================================================

async function startCamera() {

    const stream =
        await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: "environment"
            },
            audio: false
        });

    video.srcObject = stream;

    video.onloadedmetadata = () => {

        video.play();

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;

        initBtn.style.display = "none";

        statusText.innerText =
            "AI AKTIF";

        requestAnimationFrame(processFrame);
    };
}

// ======================================================
// MAIN LOOP
// ======================================================

async function processFrame() {

    if (!session) return;

    // =========================================
    // AMBIL FRAME VIDEO
    // =========================================

    ctxProcessor.drawImage(
        video,
        0,
        0,
        TARGET_SIZE,
        TARGET_SIZE
    );

    const imageData =
        ctxProcessor.getImageData(
            0,
            0,
            TARGET_SIZE,
            TARGET_SIZE
        ).data;

    // =========================================
    // CONVERT RGB
    // =========================================

    const input = new Float32Array(
        3 * TARGET_SIZE * TARGET_SIZE
    );

    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {

        input[i] =
            imageData[i * 4] / 255.0;

        input[i + TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 1] / 255.0;

        input[i + 2 * TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 2] / 255.0;
    }

    // =========================================
    // INFERENCE AI
    // =========================================

    const tensor = new ort.Tensor(
        "float32",
        input,
        [1, 3, TARGET_SIZE, TARGET_SIZE]
    );

    const results =
        await session.run({
            [session.inputNames[0]]: tensor
        });

    const output =
        results[session.outputNames[0]].data;

    // =========================================
    // PARSE OUTPUT YOLO
    // =========================================

    const boxes = [];

    const elements = 8400;
    const numClasses = CONFIG.labels.length;

    for (let i = 0; i < elements; i++) {

        let maxScore = 0;
        let classId = 0;

        for (let c = 0; c < numClasses; c++) {

            const score =
                output[i + (4 + c) * elements];

            if (score > maxScore) {

                maxScore = score;
                classId = c;
            }
        }

        if (maxScore > CONFIG.threshold) {

            let x = output[i];
            let y = output[i + elements];
            let w = output[i + elements * 2];
            let h = output[i + elements * 3];

            if (w <= 1.5) {

                x *= TARGET_SIZE;
                y *= TARGET_SIZE;
                w *= TARGET_SIZE;
                h *= TARGET_SIZE;
            }

            boxes.push({
                x: x - w / 2,
                y: y - h / 2,
                w,
                h,
                score: maxScore,
                classId
            });
        }
    }

    // =========================================
    // NMS
    // =========================================

    const finalBoxes =
        nonMaxSuppression(
            boxes,
            CONFIG.iouThreshold
        );

    drawBoxes(finalBoxes);

    requestAnimationFrame(processFrame);
}

// ======================================================
// DRAW BOX
// ======================================================

function drawBoxes(boxes) {

    ctxOverlay.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    boxes.forEach(box => {

        const scaleX =
            overlay.width / TARGET_SIZE;

        const scaleY =
            overlay.height / TARGET_SIZE;

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        ctxOverlay.strokeStyle =
            "#34C759";

        ctxOverlay.lineWidth = 3;

        ctxOverlay.strokeRect(
            x,
            y,
            w,
            h
        );

        ctxOverlay.fillStyle =
            "#34C759";

        ctxOverlay.font =
            "bold 18px Arial";

        ctxOverlay.fillText(
            `${CONFIG.labels[box.classId]} ${(box.score * 100).toFixed(1)}%`,
            x,
            y - 10
        );
    });
}

// ======================================================
// IOU
// ======================================================

function calculateIoU(box1, box2) {

    const xA =
        Math.max(box1.x, box2.x);

    const yA =
        Math.max(box1.y, box2.y);

    const xB =
        Math.min(
            box1.x + box1.w,
            box2.x + box2.w
        );

    const yB =
        Math.min(
            box1.y + box1.h,
            box2.y + box2.h
        );

    const intersection =
        Math.max(0, xB - xA) *
        Math.max(0, yB - yA);

    return intersection /
        (
            (box1.w * box1.h) +
            (box2.w * box2.h) -
            intersection
        );
}

// ======================================================
// NMS
// ======================================================

function nonMaxSuppression(
    boxes,
    iouThreshold
) {

    boxes.sort(
        (a, b) => b.score - a.score
    );

    const result = [];

    while (boxes.length > 0) {

        const current =
            boxes.shift();

        result.push(current);

        boxes =
            boxes.filter(box =>
                calculateIoU(
                    current,
                    box
                ) < iouThreshold
            );
    }

    return result;
}
