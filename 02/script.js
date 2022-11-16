/**

 * Line below creates a 2D tensor. 
 * Printing its values at this stage would give you:
 * Rank: 2, shape: [2,3], values: [[1, 2, 3], [4, 5, 6]]
 **/

 let tensor = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);


 // Create a Tensor holding a single scalar value:
 let scalar = tf.scalar(2);
 
 
 // Multiply all values in tensor by the scalar value 2.
 let newTensor = tensor.mul(scalar);
 
 
 // Values of newTensor would be: [[2, 4, 6], [8, 10 ,12]]
 newTensor.print();
 
 
 // You can even change the shape of the Tensor.
 // This would convert the 2D tensor to a 1D version with
 // 6 elements instead of a 2D 2 by 3 shape.
 let reshaped = tensor.reshape([6]);