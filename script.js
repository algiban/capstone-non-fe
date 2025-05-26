let model;
let labels = [];

// Load model dari folder
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model_h5/model.json');
    console.log("Model berhasil dimuat.");
}

// Load labels.txt
async function loadLabels() {
    const response = await fetch('labels.txt');
    const text = await response.text();
    labels = text.trim().split('\n');
    console.log("Label berhasil dimuat:", labels);
}

// Preprocess image ke tensor 224x224
function preprocessImage(image) {
    return tf.tidy(() => {
        return tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims(); // [1, 224, 224, 3]
    });
}

// Saat gambar diupload
document.getElementById('imageUpload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const img = new Image();
    const reader = new FileReader();

    reader.onload = function () {
        img.src = reader.result;
        img.onload = async function () {
            const inputTensor = preprocessImage(img);
            const prediction = await model.predict(inputTensor);
            prediction.print(); // tampilkan hasil tensor

            const index = prediction.argMax(-1).dataSync()[0];
            console.log("Prediksi Kelas ke-", index);

            let labelText = labels[index] || 'Label tidak ditemukan';
            document.getElementById('result').innerText = `Prediksi: ${labelText}`;
        };
    };

    if (file) {
        reader.readAsDataURL(file);
    }
});

// Load model dan label saat halaman dimuat
loadModel();
loadLabels();
