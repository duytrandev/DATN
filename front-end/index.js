const submitButton = document.querySelector(".btn-submit");
const categoryElement = document.getElementById("label");
const confidenceElement = document.getElementById("cond");
const loaderElement = document.getElementById("loader");

submitButton.addEventListener("click", classify);

function classify() {
  submitButton.classList.add("hide-button");
  loaderElement.classList.add("show-loader");
  fetch("https://jsonplaceholder.typicode.com/todos/1", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      categoryElement.textContent = data.title;
      confidenceElement.textContent = data.id;
      submitButton.classList.remove("hide-button");
      loaderElement.classList.remove("show-loader");
    })
    .catch((error) => {
      console.error("Error:", error);
      submitButton.classList.remove("hide-button");
      loaderElement.classList.remove("show-loader");
    });
}
