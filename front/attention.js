
// on windows load get query parameter
window.onload = function () {
    const data_field = document.getElementById('sentence');
    const urlParams = new URLSearchParams(window.location.search);
    const sentence = urlParams.get('sentence');

    data_field.innerHTML = sentence;
}
