const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';
let model = undefined;

async function loadModel() {
    model = await tf.loadLayersModel(MODEL_PATH);
    model.summary();

    // Create a batch of 1.
    const input = tf.tensor2d([[870]]);

    // Create a batch of 3
    const inputBatch = tf.tensor2d([[500], [1100], [970]]);

    // Actually make the predictions for each batch.
    const result = model.predict(input);
    const resultBatch = model.predict(inputBatch);

    // Print results to console.
    result.print();  // Or use .arraySync() to get results back as array.
    resultBatch.print(); // Or use .arraySync() to get results back as array.

    input.dispose();
    inputBatch.dispose();
    result.dispose();
    resultBatch.dispose();
    model.dispose();
}

loadModel();