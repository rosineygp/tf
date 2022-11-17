import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js';

// Input feature pairs (House size, Number of Bedrooms)
const INPUTS = TRAINING_DATA.inputs;

// Current listed house prices in dollars given their features above 
// (target output values you want to predict).
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array of Arrays needs 2D tensor to store.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output can stay 1 dimensional.
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);



// Function to take a Tensor and normalize values

// with respect to each column of values contained in that Tensor.

function normalize(tensor, min, max) {

  return tf.tidy(function () {

    // Find the minimum value contained in the Tensor.
    const MIN_VALUES = min || tf.min(tensor, 0);

    // Find the maximum value contained in the Tensor.
    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now subtract the MIN_VALUE from every value in the Tensor
    // And store the results in a new Tensor.
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // Calculate the adjusted values divided by the range size as a new Tensor.
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

}


// Normalize all input feature arrays and then 
// dispose of the original non normalized Tensors.
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log('Normalized Values:');
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log('Min Values:');
FEATURE_RESULTS.MIN_VALUES.print();

console.log('Max Values:');
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();


const model = tf.sequential()

model.add(tf.layers.dense({ inputShape: [2], units: 1 }));

model.summary()

async function train() {
  const LEARNING_RATE = 0.01; // Choose a learning rate that’s suitable for the data we are using.

  // Compile the model with the defined learning rate and specify a loss function to use.
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
  });


  // Finally do the training itself.
  let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    validationSplit: 0.15, // Take aside 15% of the data to use for validation testing.
    shuffle: true,         // Ensure data is shuffled in case it was in an order
    batchSize: 64,         // As we have a lot of training data, batch size is set to 64.
    epochs: 10             // Go over the data 10 times!
  });



  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
  console.log("Average validation error loss: " +
    Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));

  evaluate(); // Once trained evaluate the model.

}

function evaluate() {

  // Predict answer for a single piece of data.
  tf.tidy(function () {
    let newInput = normalize(tf.tensor2d([[750, 1]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);

    let output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });



  // Finally when you no longer need to make any more predictions,
  // clean up the remaining Tensors. 
  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
  console.log("done evaluate")

}

train()

// await model.save("downloads://my-model")
