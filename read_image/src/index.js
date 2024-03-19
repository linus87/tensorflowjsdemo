// Tiny TFJS train / predict example.

let fromBlobImgElement = document.getElementById('fromBlobImg');

async function loadImage(imageUrl) {
  return fetch(imageUrl)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.blob();
    })
    .then(blob => {
      return blob;
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

async function run() {
  const imageBlob = await loadImage('./cat.jpeg');
  console.log(imageBlob) ;
  fromBlobImgElement.src = URL.createObjectURL(imageBlob);

  compressImage(imageBlob, 300, 300).then(({blob, rgbTensor}) => {
    console.log(blob);
    console.log(rgbTensor);
    const img = document.getElementById('toBlobImg');
    img.src = URL.createObjectURL(blob);
  });
}
  
  run();