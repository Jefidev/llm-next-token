async function getNextToken(sentence, k) {
    try {
        const response = await fetch('http://localhost:8000/next-token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sentence: sentence, k: k })
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}


let typingTimeout;

document.getElementById('input_token').addEventListener('input', function () {
    clearTimeout(typingTimeout); // Annule le précédent timeout si l'utilisateur continue de taper

    const sentence = document.getElementById('input_token').value;
    const wordCount = sentence.split(/\s+/).filter(word => word.length > 0).length;
    const explore_link = document.getElementById('explore');

    if (wordCount > 3) {
        explore_link.classList.remove('d-none');
        typingTimeout = setTimeout(() => {
            const k = 5;
            getNextToken(sentence, k)
                .then(data => {
                    document.getElementById('output_token').innerHTML = '';
                    for (let key in data) {
                        let value = data[key].toFixed(4);
                        let percentage = (value * 100) + 10;
                        let jauge_value = percentage.toFixed(2);
                        document.getElementById('output_token').innerHTML +=
                            `<div class="key_jauge"> 
                                <span id='key' onclick='addtext(this)'>${key}</span> :  
                                <div class="progress-container">
                                    <div class="progress">
                                        <div class="progress-bar" role="progressbar" 
                                            style="width:${percentage}%" aria-valuemin="0" aria-valuemax="10"> 
                                            ${jauge_value}%
                                        </div>
                                    </div>
                                </div>
                            </div>`;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }, 1500);
    } else {
        explore_link.classList.add('d-none');
        document.getElementById('output_token').innerHTML = '';
    }
});


function addtext(element) {
    let text_area = document.getElementById('input_token');
    text_area.value += '' + element.innerHTML;

    text_area.dispatchEvent(new Event('input'));
}

