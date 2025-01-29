let attention_score;
let attention_idx = 0;
let lastScrollTime = 0; // To track the last event time
const scrollThreshold = 100; // Time in milliseconds
const gestureThreshold = 50; // Minimum delta for meaningful gesture

const color_ranges = [
    [[173, 216, 230], [0, 191, 255]],
    [[144, 238, 144], [152, 251, 152]],
    [[255, 54, 58], [255, 182, 193]],
    [[255, 165, 0], [255, 255, 224]],
    [[211, 91, 239], [238, 174, 255]]
]

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

async function displayAttentionSvg(sentence, idx) {
    try {
        const response = await fetch('http://localhost:8000/attention-plot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sentence: sentence, idx: idx })
        });

        const data = await response;
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}


function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

function displayAttentionScore(attention_score, sentence) {
    const attention_score_div = document.getElementById('attention');
    const tokens = attention_score["tokens"];

    // Change Label
    const label = document.getElementById('head-tag');
    label.innerHTML = `Head ${attention_idx + 1}`;

    displayAttentionSvg(sentence, attention_idx).then(response => response.text())
        .then(svg => {
            document.getElementById('svg-container').innerHTML = svg;
        })

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

        // generate random color
        selected_color = color_ranges[attention_idx % color_ranges.length];

        start_rgb = selected_color[0];
        end_rgb = selected_color[1];

        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * weight
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * weight
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * weight

        let color = `rgba(${r}, ${g}, ${b}, ${weight})`;
        generated_html += `<span class='soft-btn btn-outline-dark display-6' style="background-color: ${color}; padding: 2px; margin: 2px; border-radius: 4px;">${token}</span> `;
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
            displayAttentionScore(attention_score, sentence);
        });

}

window.addEventListener('wheel', (event) => {
    const currentTime = Date.now();
    const timeDiff = currentTime - lastScrollTime;
    const urlParams = new URLSearchParams(window.location.search);
    const sentence = urlParams.get('sentence');

    if (timeDiff > scrollThreshold && Math.abs(event.deltaY) > gestureThreshold) {

        dic_size = Object.keys(attention_score["heads"]).length;

        if (event.deltaY > 0) {
            attention_idx = Math.min(dic_size - 1, attention_idx + 1);
        } else {
            attention_idx = Math.max(0, attention_idx - 1);
        }

        displayAttentionScore(attention_score, sentence);
    }

    lastScrollTime = currentTime; // Update the last scroll time
});
