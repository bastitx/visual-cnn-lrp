//Global Variables
var customBoard, tinyCtx
var renderer, camera, scene
var nnMetaData, nnOutput
var heatmap_hide = false
var selectedMethod = "Epsilon"
var selectedModel = "Linear"
imagesize = 28
SERVER="http://127.0.0.1:5000" // where backend API is located


/*
Called when webpage is initialized
*/
function init() {
  //Initialize Drawing Board
  //See https://github.com/Leimi/drawingboard.js/ for documentation
  customBoard = new DrawingBoard.Board('custom-board', {
    background: "#000",
    color: "#fff",
    size: 30,
    controls: [
      { Navigation: { back: false, forward: false } },
      { DrawingMode: { filler: false } }
    ],
    controlsPosition: "bottom right",
    webStorage: 'session',
    droppable: false
  });
  //initialize scaled down version of drawing, which is input of neural network
  tinyCtx = $("#tiny")[0].getContext("2d");
  tinyCtx.scale(imagesize/280,imagesize/280);

  // Initialize 3D scene
  // See documentation https://threejs.org/docs
  scene = new THREE.Scene();
  var aspect = window.innerWidth / window.innerHeight;
  camera = new THREE.OrthographicCamera( -680*aspect, 680*aspect, 600, -760, -100, 50 );

  renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true} );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.setPixelRatio( window.devicePixelRatio );
  document.getElementById("webgl_container").appendChild( renderer.domElement );

  var controls = new THREE.OrbitControls( camera, renderer.domElement );
  window.addEventListener( 'resize', onWindowResize, false );
  // Initialize the heatmap hide button, which switches between showing
  // activations and heatmap
  $('#heatmap_hide_button').click(function() {
					heatmap_hide = !heatmap_hide;
          updateNetwork()
  });
  
  setModel(selectedModel)
}

function setModel(model) {
  // Initialize the "cubes"(in 2d just squares)
  // These are the building blocks of the visualization of the neural network
  // Each cube represents a neuron
  // Metadata of neural network is pulled from backend
  selectedModel = model
  customBoard.ev.unbind('board:stopDrawing', updateNetwork)
  $.get(SERVER+"/metaData/"+selectedModel).done(function(result) {
    nnMetaData = result
    drawCubes()
    customBoard.ev.bind('board:stopDrawing', updateNetwork);
    $('#modelselection')[0].textContent = model
  });
}

// Function that's called right after someone has just stopped drawing in the
// DrawingBoard. This will send the drawn image to the neural network
function updateNetwork() {
  // Get the content of the image, scale it down, and store it in tinyCtx
  var imageData = customBoard.canvas.getContext("2d").getImageData(0,0,imagesize*10,imagesize*10)
  var newCanvas = $("<canvas>").attr("width", imageData.width).attr("height", imageData.height)[0];
  newCanvas.getContext("2d").putImageData(imageData, 0, 0);
  tinyCtx.drawImage(newCanvas, 0, 0);
  imageDataScaled = tinyCtx.getImageData(0,0,imagesize,imagesize).data //get RGBA array of image
  //Save alpha value of RGBA image and scale it from [0, 255] to [0, 1]
  var input = new Array(imagesize*imagesize)
  for(i=0; i<input.length; i++) {
    //input[i] = imageDataScaled[i*4]/255 //scale [0,1]

    //Scale to [-1,1]
    input[i] = imageDataScaled[i*4]/127.5-1 //TODO make this configurable
  }
  input = math.reshape(input, [1,1,imagesize,imagesize])
  //Parse heatmap selector, which can override the
  heatmap_selector = parseInt($('#ans3')[0].value)
  heatmap_selector = 0 <= heatmap_selector < 10 ? heatmap_selector : -1
  lrp_parameter = parseFloat($('#lrpparam')[0].value)
  input_data = {
    data: input,
    heatmap_selection: heatmap_selector,
    parameter: lrp_parameter
  }
  url = SERVER+'/'
  url += heatmap_hide ? 'activations/'+selectedModel : 'lrp/'+selectedModel+'/'+selectedMethod
  $.ajax({
    type: 'post',
    url: url,
    contentType: 'application/json',
    data: JSON.stringify(input_data),
    success: function(data) {
      nnOutput = data
      updateCubes(data)
      var output = JSON.parse(JSON.stringify(data[data.length-1][0].flat(2)))
      ans1 = argMax(output)
      output[ans1] = -9999999999
      ans2 = argMax(output)
      document.getElementById("ans1").innerHTML = ans1
      document.getElementById("ans2").innerHTML = ans2
    },
    error: function( jqXhr, textStatus, errorThrown ){
      console.log( errorThrown );
    }
  })
}

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function onWindowResize( e ) {
  var aspect = window.innerWidth / window.innerHeight;
  camera.left = -680*aspect;
  camera.right = 680*aspect;
  camera.top = 600;
  camera.bottom = -760;
  camera.updateProjectionMatrix();

  renderer.setSize( window.innerWidth, window.innerHeight );
  render()
}


function updateCubes(data) {
  data_ = data.flat(4)
  data_rel = data[0].flat(3)
  color_scale = chroma.scale('RdBu')
  d = math.max(math.abs(data_rel))
  color_scale_ = color_scale.domain([-d, d])
  for(i=0; i<data_.length; i++) {
    var color = new THREE.Color(color_scale_(data_[i]).num())
    for(j=0; j<12; j++) {
      for(k=0; k<3; k++) {
        scene.children[0].geometry.faces[i*12+j].vertexColors[k] = color
      }
    }
  }
  scene.children[0].geometry.colorsNeedUpdate = true;
  scene.children[0].geometry.verticesNeedUpdate = true;
  
  var imageData = customBoard.canvas.getContext("2d").getImageData(0,0,280,280)
  var outputsize = 140
  var img_u8_big = new jsfeat.matrix_t(280, 280, jsfeat.U8C1_t);
  var img_u8 = new jsfeat.matrix_t(outputsize, outputsize, jsfeat.U8C1_t);
  jsfeat.imgproc.grayscale(imageData.data, 280, 280, img_u8_big)
  jsfeat.imgproc.resample(img_u8_big, img_u8, outputsize, outputsize)
  jsfeat.imgproc.gaussian_blur(img_u8, img_u8, 2, 0)
  jsfeat.imgproc.canny(img_u8, img_u8, 0, 0)

  var img_c4 = Array.from(img_u8.data).flatMap((x) => [x,x,x,255])
  var img_final = new ImageData(new Uint8ClampedArray(img_c4), outputsize, outputsize)
  var texture = new THREE.Texture(img_final)
  scene.children[1].material.alphaMap = texture
  scene.children[1].material.alphaMap.needsUpdate = true
}

function drawCubes() {
  for(i=scene.children.length-1; i>=0; i--)
    scene.remove(scene.children[i]);
  var height = -600
  var geometry = new THREE.Geometry();
  //var defaultMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff, shading: THREE.FlatShading, vertexColors: THREE.VertexColors, transparent: true} );
  var defaultMaterial = new THREE.MeshBasicMaterial({vertexColors: THREE.VertexColors});
  var geom = new THREE.BoxGeometry( 9, 9, 9 );
	var quaternion = new THREE.Quaternion();
  quaternion.setFromEuler( new THREE.Euler(0,0,0), false );
	var scale = new THREE.Vector3(1,1,1);
  for(h=0; h<nnMetaData.length; h++) {
    layer = nnMetaData[h]
    if(layer.outputsize.length == 2) {
      layer.outputsize = [1,1].concat(layer.outputsize)
    }
    height += layer.outputsize[2]*10+20
    drawLayer(layer, height, quaternion, scale, geom, geometry)
  }
  var drawnObject = new THREE.Mesh( geometry, defaultMaterial);
	drawnObject.name = 'cubes';
  scene.add( drawnObject );
  
  var texture = new THREE.Texture(customBoard.canvas)
  geometry = new THREE.BoxGeometry(nnMetaData[0].outputsize[3]*10, nnMetaData[0].outputsize[2]*10, 1);
  var imageMaterial = new THREE.MeshBasicMaterial({map: texture, alphaMap:texture, transparent: true})
  var object = new THREE.Mesh(geometry, imageMaterial)
  height = -600 + (nnMetaData[0].outputsize[2]*10)/2 + 25
  object.position.set(-15, height, 10)
  scene.add(object)
}

function drawLayer(layer, height, quaternion, scale, geom, geometry) {
  var width=-layer.outputsize[1]*(layer.outputsize[3]*10+20)/2
  for(i=0; i<layer.outputsize[1]; i++) {
    for(j=0; j<layer.outputsize[2]; j++) {
      for(k=0; k<layer.outputsize[3]; k++) {
        var position = new THREE.Vector3(width+k*10, height-j*10, 0);
        var matrix = new THREE.Matrix4();
        matrix.compose(position, quaternion, scale);
        var color = new THREE.Color(0,0,0);
        geom.faces.forEach(function(face) {
          for(l=0; l<3; l++) {
            face.vertexColors[l] = color
          }
        });
        geometry.merge(geom, matrix);
      }
    }
    width += layer.outputsize[3]*10+20
  }
}

function animate() {
  requestAnimationFrame( animate );
  render();
}

function render() {
  renderer.render(scene, camera);
}

function changeMethod(method) {
  selectedMethod = method;
  updateNetwork()
  $('#lrpselection')[0].textContent = method
}

init();
animate();
