let attention_score;
let attention_idx = 0;
let lastScrollTime = 0; // To track the last event time
const scrollThreshold = 100; // Time in milliseconds
const gestureThreshold = 50; // Minimum delta for meaningful gesture

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

function displayAttentionScore(attention_score) {
    const attention_score_div = document.getElementById('attention');
    const tokens = attention_score["tokens"];

    // Change Label
    const label = document.getElementById('head-tag');
    label.innerHTML = `Head ${attention_idx + 1}`;

    // get key at index idx in the dictionary
    const keys = Object.keys(attention_score["heads"]);
    const key = keys[attention_idx];

    // get the value of the key
    const value = attention_score["heads"][key];

    let generated_html = ``;
    let zipped_list = tokens.map((token, i) => [token, value[i]]);

    for (let i = 0; i < zipped_list.length; i++) {
        let token = zipped_list[i][0];
        let weight = zipped_list[i][1];

        let color = `rgba(255, 165, 0, ${weight})`;
        generated_html += `<span style="background-color: ${color}; padding: 2px; margin: 2px; border-radius: 4px;">${token}</span> `;
    }

    attention_score_div.innerHTML = generated_html;

}


// on windows load get query parameter
window.onload = function () {
    const urlParams = new URLSearchParams(window.location.search);
    const sentence = urlParams.get('sentence');

    // Get the attention score
    getAttentionHead(sentence)
        .then(data => {
            attention_score = data;
            console.log(attention_score);
            displayAttentionScore(attention_score);
        });

}

window.addEventListener('wheel', (event) => {
    const currentTime = Date.now();
    const timeDiff = currentTime - lastScrollTime;

    if (timeDiff > scrollThreshold && Math.abs(event.deltaY) > gestureThreshold) {

        dic_size = Object.keys(attention_score["heads"]).length;

        if (event.deltaY > 0) {
            attention_idx = Math.min(dic_size - 1, attention_idx + 1);
        } else {
            attention_idx = Math.max(0, attention_idx - 1);
        }

        displayAttentionScore(attention_score);

    }

    lastScrollTime = currentTime; // Update the last scroll time
});
