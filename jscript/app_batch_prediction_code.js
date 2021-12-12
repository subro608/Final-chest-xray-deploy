





//################################################################################

// ### 1. MAKE A PREDICTION ON THE IMAGE OR MULTIPLE IMAGES THAT THE USER SUBMITS

//#################################################################################
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
      
      for (let i = 1;i<2;i++){
        ge_curr = ge_prev + lam * fmlambda[i] * ge_prev;
        fuzzy_pred = Math.max(fuzzy_pred,Math.min(scoreslambda[i],ge_curr));
        ge_prev = ge_curr;
      }
      fuzzy_pred = Math.max(fuzzy_pred,Math.min(scoreslambda[2],1));
      console.log(fuzzy_pred);
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
      
      for (let i = 1;i<2;i++){
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

// the model images have size 96x96

async function model_makePrediction(fname) {
	
	//console.log('met_cancer');
	
	// clear the previous variable from memory.
	let image = undefined;
	let model;
  let model3;
  let model4;
	image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();
  
  let grayscale = tensor.mean(3).expandDims(-1);
	model = await tf.loadModel('./model_kaggle_1/model.json');
  model3 = await tf.loadModel('./model_kaggle_1/model 3/model.json');
  model4 = await tf.loadModel('./model_kaggle_1/model 4/model.json');
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	let predictions = await model.predict(tensor).data();
  let predictions3 = await model3.predict(grayscale).data();
  let predictions4 = await model4.predict(tensor).data();
  let sug = sugeno([predictions],[predictions3],[predictions4]);
  let cho = choquet([predictions],[predictions3],[predictions4]);
  
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] // we are selecting the value from the obj
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
	let top53 = Array.from(predictions3)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] // we are selecting the value from the obj
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
  
	let top54 = Array.from(predictions4)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] // we are selecting the value from the obj
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
  
	let top_sug = Array.from(sug)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
  
	let top_cho = Array.from(cho)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
		
	// Append the file name to the prediction list
	$("#prediction-list").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">
	${fname}</li>`);
	$("#prediction-list1").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">
	${fname}</li>`);
  $("#prediction-list2").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">
	${fname}</li>`);
  $("#prediction-list3").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">
	${fname}</li>`);
  $("#prediction-list4").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">
	${fname}</li>`);
	
	//$("#prediction-list").empty();
	top5.forEach(function (p) {
		$("#prediction-list").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
	});
	// Add a space after the prediction for each image
	$("#prediction-list").append(`<br>`);
  
	top53.forEach(function (p) {
		$("#prediction-list1").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
	});
	// Add a space after the prediction for each image
	$("#prediction-list1").append(`<br>`);
  
  
	top54.forEach(function (p) {
		$("#prediction-list2").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
	});
	// Add a space after the prediction for each image
	$("#prediction-list2").append(`<br>`);
  
	top_sug.forEach(function (p) {
		$("#prediction-list3").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
	});
	// Add a space after the prediction for each image
	$("#prediction-list3").append(`<br>`);

	top_cho.forEach(function (p) {
		$("#prediction-list4").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
	});
	// Add a space after the prediction for each image
	$("#prediction-list4").append(`<br>`);
		
}




// =====================
// The following functions help to solve the problems relating to delays 
// in assigning the src attribute and the delay in model prediction.
// Without this the model will produce unstable predictions because
// it will not be predicting on the correct images.


// This tutorial explains how to use async, await and promises to manage delays.
// Tutorial: https://blog.lavrton.com/javascript-loops-how-to-handle-async-await-6252dd3c795
// =====================



function model_delay() {
	
	return new Promise(resolve => setTimeout(resolve, 200));
}


async function model_delayedLog(item, dataURL) {
	
	// We can await a function that returns a promise.
	// This delays the predictions from appearing.
	// Here it does not actually serve a purpose.
	// It's here to show how a delay like this can be implemented.
	await model_delay();
	
	// display the user submitted image on the page by changing the src attribute.
	// The problem is here. Too slow.
	$("#selected-image").attr("src", dataURL);
	$("#displayed-image").attr("src", dataURL); //#########
	
	// log the item only after a delay.
	//console.log(item);
}

// This step by step tutorial explains how to use FileReader.
// Tutorial: http://tutorials.jenkov.com/html5/file-api.html

async function model_processArray(array) {
	
	for(var item of fileList) {
		
		
		let reader = new FileReader();
		
		// clear the previous variable from memory.
		let file = undefined;
	
		
		reader.onload = async function () {
			
			let dataURL = reader.result;
			
			await model_delayedLog(item, dataURL);
			
			
			
			var fname = file.name;
			
			// clear the previous predictions
			$("#prediction-list").empty();
      $("#prediction-list1").empty();
      $("#prediction-list2").empty();
      $("#prediction-list3").empty();
      $("#prediction-list4").empty();
			
			// 'await' is very important here.
			await model_makePrediction(fname);
		}
		
		file = item;
		
		// Print the name of the file to the console
        //console.log("i: " + " - " + file.name);
			
		reader.readAsDataURL(file);
	}
}













