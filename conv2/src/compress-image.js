/**
 * Compress a image with canvas.
 * @param {} file 
 * @returns 
 */
async function compressImage(blob, desiredWidth, desiredHeight) {
  const img = new Image();
  const newImageWidth = desiredWidth ?? 300; // For example, 300 pixels width
  const newImageHeight = desiredHeight ?? 300; // For example, 300 pixels height

  return new Promise((resolve, reject) => {
    img.onload = function() {
      // Set the desired width and height
      
      const canvas = document.getElementById('image-canvas');
      const ctx = canvas.getContext('2d');

      // Calculate the scaling factor to keep the aspect ratio
      // let width = img.width;
      // let height = img.height;
      // const scalingFactor = Math.min(maxWidth / width, maxHeight / height);
      // width *= scalingFactor;
      // height *= scalingFactor;

      // Set the canvas size to the new dimensions
      canvas.width = newImageWidth;
      canvas.height = newImageHeight;

      // Draw the image onto the canvas
      ctx.drawImage(img, 0, 0, newImageWidth, newImageHeight);

      let iamgeBlob;

      canvas.toBlob(blob => {
        iamgeBlob = blob;
      }, 'image/jpeg', 0.8);

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

      // Convert the canvas content to a data URL (base64 encoded image)
      canvas.toBlob(blob => {
          resolve({blob, rgbTensor});
      }, 'image/jpeg', 0.8);
    };
    img.src = URL.createObjectURL(blob);;
  });
    
}