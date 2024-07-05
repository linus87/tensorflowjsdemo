export default async function loadImage(imageUrl: string) {
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