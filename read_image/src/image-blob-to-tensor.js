
async function imageBlobToTensor(blob) {
  const img = new Image();

  return new Promise((resolve, reject) => {
    img.onload = function() {
      const canvas = document.getElementById('image-canvas');
      const ctx = canvas.getContext('2d');
    
      // Draw the image onto the canvas
      ctx.drawImage(img, 0, 0);
      // Get the image data from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const data = imageData.data;
    
      // Create the RGB tensor (3D array)
      const rgbTensor = [];
      for (let y = 0; y < canvas.height; y++) {
          const row = [];
          for (let x = 0; x < canvas.width; x++) {
              const idx = (y * canvas.width + x) * 4;
              const pixel = [
                  data[idx],     // Red channel
                  data[idx + 1], // Green channel
                  data[idx + 2]  // Blue channel
                  // We're ignoring the alpha channel (idx + 3)
              ];
              row.push(pixel);
          }
          rgbTensor.push(row);
      }
  
      // Now you have the RGB tensor, you can use it as needed
      resolve(rgbTensor);
    };
    
    img.src = URL.createObjectURL(blob);;
  });
}
