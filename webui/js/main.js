//Global Variables
var customBoard, tinyCtx
var renderer, camera, scene
var nnMetaData, nnOutput
var heatmap_hide = false

function init() {
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

  tinyCtx = $("#tiny")[0].getContext("2d");
  tinyCtx.scale(0.1,0.1);

  scene = new THREE.Scene();
  var aspect = window.innerWidth / window.innerHeight;
  camera = new THREE.OrthographicCamera( -680*aspect, 680*aspect, 600, -760, -100, 50 );

  renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true} );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.setPixelRatio( window.devicePixelRatio );
  document.getElementById("webgl_container").appendChild( renderer.domElement );

  var controls = new THREE.OrbitControls( camera, renderer.domElement );
  window.addEventListener( 'resize', onWindowResize, false );
  $.get("http://127.0.0.1:5000/metaData").done(function(result) {
      nnMetaData = result
      drawCubes()
      customBoard.ev.bind('board:stopDrawing', onStopDrawing);
  });

  $('#heatmap_hide_button').click(function() {
					heatmap_hide = !heatmap_hide;
          onStopDrawing()
	});
}

function onStopDrawing() {
  var imageData = customBoard.canvas.getContext("2d").getImageData(0,0,280,280)
  var newCanvas = $("<canvas>").attr("width", imageData.width).attr("height", imageData.height)[0];
  newCanvas.getContext("2d").putImageData(imageData, 0, 0);
  tinyCtx.drawImage(newCanvas, 0, 0);
  imageDataScaled = tinyCtx.getImageData(0,0,28,28).data
  var input = new Array(28*28)
  for(i=0; i<input.length; i++) {
    input[i] = imageDataScaled[i*4]/255
  }
  input = math.reshape(input, [1,1,28,28])
  heatmap_selector = parseInt($('#ans3')[0].value)
  heatmap_selector = 0 <= heatmap_selector < 10 ? heatmap_selector : -1
  input_data = {
    data: input,
    heatmap_selection: heatmap_selector
  }
  url = 'http://127.0.0.1:5000/'
  url += heatmap_hide ? 'activations' : 'lrp/alphabeta'
  $.ajax({
    type: 'post',
    url: url,
    contentType: 'application/json',
    data: JSON.stringify(input_data),
    success: function(data) {
      nnOutput = data
      updateCubes(data)
      var output = JSON.parse(JSON.stringify(data[data.length-1][0]))
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
  color_scale = chroma.scale('RdBu')
  distinct_data = [...new Set(data_)]
  distinct_data_n = distinct_data.filter((x) => x <= 0)
  distinct_data_p = distinct_data.filter((x) => x >= 0)
  //quantiles = math.quantileSeq([...new Set(data_)], color_scale.colors().length)
  quantiles_n = math.quantileSeq(distinct_data_n, 6).slice(0, 5)
  quantiles_p = math.quantileSeq(distinct_data_p, 6).slice(1, 6)
  color_scale_ = color_scale.domain([quantiles_n, 0, quantiles_p].flat())
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
}

function drawCubes() {
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

init();
animate();
