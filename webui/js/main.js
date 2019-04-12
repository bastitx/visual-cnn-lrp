//Global Variables
var customBoard, renderer, camera, scene

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

  scene = new THREE.Scene();
  var aspect = window.innerWidth / window.innerHeight;
  camera = new THREE.OrthographicCamera( -680*aspect, 680*aspect, 600, -760, -100, 50 );

  renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true} );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.setPixelRatio( window.devicePixelRatio );
  document.getElementById("webgl_container").appendChild( renderer.domElement );

  var geometry = new THREE.BoxGeometry( 200, 200, 200 );
  var material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
  var cube = new THREE.Mesh( geometry, material );
  scene.add( cube );

  window.addEventListener( 'resize', onWindowResize, false );

}

function onWindowResize( e ) {
  var aspect = window.innerWidth / window.innerHeight;
  camera.left = -680*aspect;
  camera.right = 680*aspect;
  camera.top = 600;
  camera.bottom = -760;
  camera.updateProjectionMatrix();

  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.render();
}

function animate() {
  requestAnimationFrame( animate );
  renderer.render(scene, camera);
}

init();
animate();
