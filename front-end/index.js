const submitButton = document.querySelector(".btn-submit");
const categoryElement = document.getElementById("label");
const confidenceElement = document.getElementById("cond");
const loaderElement = document.getElementById("loader");
const selectItem = document.querySelector(".input-select");
const result = document.querySelector(".label-choose");
const content = document.querySelector(".input-text-area");
submitButton.addEventListener("click", classify);
selectItem.addEventListener("change", (e) => {
  categoryElement.textContent = "";
  confidenceElement.textContent = "";
});

function classify(e) {
  submitButton.classList.add("hide-button");
  loaderElement.classList.add("show-loader");
  const algorithm = selectItem.value;
  const bodyInput = { input: content.value };
  fetch(`http://127.0.0.1:8000/predict/${algorithm}/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(bodyInput),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      categoryElement.textContent = data.label;
      confidenceElement.textContent = data.conf;
      submitButton.classList.remove("hide-button");
      loaderElement.classList.remove("show-loader");
    })
    .catch((error) => {
      console.error("Error:", error);
      submitButton.classList.remove("hide-button");
      loaderElement.classList.remove("show-loader");
    });
}
