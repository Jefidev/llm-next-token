let attention_score;

async function getAttentionHead(sentence) {
    try {
        const response = await fetch('http://localhost:8000/attention-score', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sentence: sentence })
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}


// on windows load get query parameter
window.onload = function () {
    const data_field = document.getElementById('sentence');
    const urlParams = new URLSearchParams(window.location.search);
    const sentence = urlParams.get('sentence');

    data_field.innerHTML = sentence;

    // Get the attention score
    getAttentionHead(sentence)
        .then(data => {
            attention_score = data;
            console.log(attention_score);
        });

}
