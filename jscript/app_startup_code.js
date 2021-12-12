


//#############################################################

// ### 1. LOAD THE MODEL IMMEDIATELY WHEN THE PAGE LOADS

//#############################################################

// Define 2 helper functions

function simulateClick(tabID) {
	
	document.getElementById(tabID).click();
}

function sort(a,b){
  const array_len = a.length;
  const index = new Array(array_len);
  const new_array = new Array(array_len).fill(0);
  for (var j = 0; j < array_len;j++){
    new_array[j] = a[b[j]];
}   
  return new_array;
}

function sugeno(mobilenet_test,resnet_test,densenet_test){
  var fuzzy_measure = [0.3659,0.3012, 0.1837];
  const zeros = (m, n) => [...Array(m)].map(e => Array(n).fill(0));
  const lam = 0.6084;
  const class_num = 3;
  const sample_num = 1;
  
  var ypred_fuzzy = zeros(sample_num,3);
  
  for (let sample = 0; sample<sample_num; sample++){
    for (let classes = 0; classes <class_num;classes++){
      var scores = [mobilenet_test[sample][classes],resnet_test[sample][classes],densenet_test[sample][classes]];
      var scoreslambda = scores.sort((a, b) => b-a);
      var len = scores.length;
      var indices = new Array(len);
      for (var i = 0; i < len; ++i) indices[i] = i;
      indices.sort(function (a, b) { return scores[a] > scores[b] ? -1 : scores[a] < scores[b] ? 1 : 0;});
      var fmlambda = sort(fuzzy_measure,indices);

      var ge_curr;
      var ge_prev = fmlambda[0];
      var fuzzy_pred = Math.min(scoreslambda[0], fmlambda[0]);
      
      for (let i = 1;i<3;i++){
        ge_curr = ge_prev + lam * fmlambda[i] * ge_prev;
        fuzzy_pred = Math.max(fuzzy_pred,Math.min(scoreslambda[i],ge_curr));
        ge_prev = ge_curr;
      }
      fuzzy_pred = Math.max(fuzzy_pred,Math.min(scoreslambda[2],1));
      ypred_fuzzy[sample][classes] = fuzzy_pred;
}
} console.log(normalize(ypred_fuzzy[0],ypred_fuzzy[0].reduce(add,0)));
  return normalize(ypred_fuzzy[0],ypred_fuzzy[0].reduce(add,0));
}

function choquet(mobilenet_test,resnet_test,densenet_test){
  var fuzzy_measure = [0.3659,0.3012, 0.1837];
  const zeros = (m, n) => [...Array(m)].map(e => Array(n).fill(0));
  const lam = 0.6084;
  const class_num = 3;
  const sample_num = 1;
  
  var ypred_fuzzy = zeros(sample_num,3);
  
  for (let sample = 0; sample<sample_num; sample++){
    for (let classes = 0; classes <class_num;classes++){
      var scores = [mobilenet_test[sample][classes],resnet_test[sample][classes],densenet_test[sample][classes]];
      var scoreslambda = scores.sort((a, b) => b-a);
      var len = scores.length;
      var indices = new Array(len);
      for (var i = 0; i < len; ++i) indices[i] = i;
      indices.sort(function (a, b) { return scores[a] > scores[b] ? -1 : scores[a] < scores[b] ? 1 : 0;});
      var fmlambda = sort(fuzzy_measure,indices);
      var ge_curr;

      var ge_prev = fmlambda[0];
      var fuzzy_pred = scoreslambda[0] * fmlambda[0];
      
      for (let i = 1;i<3;i++){
        ge_curr = ge_prev + lam * fmlambda[i] * ge_prev;
        fuzzy_pred = fuzzy_pred + scoreslambda[i] * (ge_curr-ge_prev);
        ge_prev = ge_curr;
      }
      fuzzy_pred = fuzzy_pred + scoreslambda[2] * (1-ge_prev);
      console.log(fuzzy_pred);
      ypred_fuzzy[sample][classes] = fuzzy_pred;
}
} console.log(normalize(ypred_fuzzy[0],ypred_fuzzy[0].reduce(add,0)));
  return normalize(ypred_fuzzy[0],ypred_fuzzy[0].reduce(add,0));
}

function add(accumulator, a) {
  return accumulator + a;
}

function normalize(arr, max) {
    // find the max value
    var m = 0;
    for(var x=0; x<arr.length; x++) m = Math.max(m, arr[x]);
    // find the ratio
    var r = max / m;
    // normalize the array
    for(var x=0; x<arr.length; x++) arr[x] = arr[x] * r;
    return arr;
}


function predictOnLoad() {
	
	// Simulate a click on the predict button
	setTimeout(simulateClick.bind(null,'predict-button'), 500);
}






// LOAD THE MODEL

let model;
let model3;
let model4;
(async function () {
	
	model = await tf.loadModel('./model_kaggle_1/model.json');
  model3 = await tf.loadModel('./model_kaggle_1/model 3/model.json');
  model4 = await tf.loadModel('./model_kaggle_1/model 4/model.json');
	$("#selected-image").attr("src", "./assets/default_image.jpeg");
	
	// Hide the model loading spinner
	// This line of html gets hidden:
	// <div class="progress-bar">Ai is Loading...</div>
	$('.progress-bar').hide();
	
	
	// Simulate a click on the predict button.
	// Make a prediction on the default front page image.
	predictOnLoad();
	
	
	
})();


	

//######################################################################

// ### 2. MAKE A PREDICTION ON THE FRONT PAGE IMAGE WHEN THE PAGE LOADS

//######################################################################



// The model images have size 96x96

// This code is triggered when the predict button is clicked i.e.
// we simulate a click on the predict button.
$("#predict-button").click(async function () {
	
	let image = undefined;
	// const tf = require('@tensorflow/tfjs');
	// const TARGET_CLASSES = require('target_classes');
	image = $('#selected-image').get(0);
	
	// Pre-process the image


	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();
  
//   let edge_tensor = tf.image.sobel_edges(tensor);
// 	edge_tensor.print();
  let grayscale = tensor.mean(3).expandDims(-1);
	
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	
	// TARGET_CLASSES is defined in the target_clssses.js file.
	// There's no need to load this file because it was imported in index.html
	let predictions = await model.predict(tensor).data();
  let predictions3 = await model3.predict(grayscale).data();
  let predictions4 = await model4.predict(tensor).data();
  
  let sug = sugeno([predictions],[predictions3],[predictions4]);
  let cho = choquet([predictions],[predictions3],[predictions4]);
//   let top_sug = [{
//   className: TARGET_CLASSES[0],
//   probability: sug[0][0]
// }, {
//   className: TARGET_CLASSES[1],
//   probability: sug[0][1]
// }, {
//   className: TARGET_CLASSES[2],
//   probability: sug[0][2]
// }];
 
  

  
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		});
	let top53 = Array.from(predictions3)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		});
	let top54 = Array.from(predictions4)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		});
  
	let top_sug = Array.from(sug)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		});
  
	let top_cho = Array.from(cho)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		});
  
	  // var float32 = new Float32Array(sug);
	  // console.log("model1");
	  // console.log(predictions4);
	  console.log("sugeno");
    console.log(sug[0]);
		// Append the file name to the prediction list
		var file_name = 'default_image.jpeg';
		$("#prediction-list").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);
		$("#prediction-list1").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);  
		$("#prediction-list2").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);		
		$("#prediction-list3").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);		
    $("#prediction-list4").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);	
  
		top5.forEach(function (p) {
			$("#prediction-list").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
		});
		top53.forEach(function (p) {
      $("#prediction-list1").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
		});
		top54.forEach(function (p) {
      $("#prediction-list2").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);	
		});
		top_sug.forEach(function (p) {
		$("#prediction-list3").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);	
		});
		top_cho.forEach(function (p) {
		$("#prediction-list4").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);	
		});

	
});



//######################################################################

// ### 3. READ THE IMAGES THAT THE USER SELECTS

// Then direct the code execution to app_batch_prediction_code.js

//######################################################################




// This listens for a change. It fires when the user submits images.

$("#image-selector").change(async function () {
  // the FileReader reads one image at a time
	fileList = $("#image-selector").prop('files');
	
	//$("#prediction-list").empty();
	
	// Start predicting
	// This function is in the app_batch_prediction_code.js file.
	model_processArray(fileList);
	
});





