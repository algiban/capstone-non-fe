let model;
let labels = [];

// Load model dari folder
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model_h5/model.json');
    console.log("Model berhasil dimuat.");
}

// Load label dari labels.txt
async function loadLabels() {
    const response = await fetch('labels.txt');
    const text = await response.text();
    labels = text.trim().split('\n');
    console.log("Label berhasil dimuat:", labels);
}

// Ambil webcam
async function setupWebcam() {
    const webcamElement = document.getElementById('webcam');

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamElement.srcObject = stream;

            return new Promise((resolve) => {
                webcamElement.onloadedmetadata = () => {
                    resolve(webcamElement);
                };
            });
        } catch (err) {
            alert("Gagal mengakses webcam: " + err.message);
            throw err; // biar tahu errornya dan tidak lanjut
        }
    } else {
        alert("Browser Anda tidak mendukung getUserMedia API.");
        throw new Error("getUserMedia not supported");
    }
}

// Fungsi prediksi loop
async function predictLoop(video) {
    while (true) {
        const input = tf.tidy(() => {
            return tf.browser.fromPixels(video)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims(); // [1, 224, 224, 3]
        });

        const prediction = await model.predict(input);
        const index = prediction.argMax(-1).dataSync()[0];
        const label = labels[index] || 'Tidak dikenali';

        document.getElementById('result').innerText = `Prediksi: ${label} (kelas ${index})`;

        await tf.nextFrame(); // Tunggu frame berikutnya
    }
}

// Inisialisasi semua
async function main() {
    await loadModel();
    await loadLabels();
    try {
        const video = await setupWebcam();
        predictLoop(video);
    } catch (err) {
        console.error("Gagal memulai webcam atau prediksi:", err);
    }
}

main();
