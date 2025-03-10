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

// Override click function
document.getElementById('explore').onclick = function () {
    const sentence = document.getElementById('input_token').value;

    // Redirect to the explore page
    window.location.href = `attention.html?sentence=${sentence}`;
}


document.getElementById('input_token').addEventListener('input', function () {
    clearTimeout(typingTimeout); 

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
                    let total = 0;
                    let percentages = [];
                    for (let key in data) {
                        let value = parseFloat(data[key].toFixed(4));  
                        total += value;
                        percentages.push({ key: key, value: value });
                    }
                    
                    percentages.forEach(item => {
                        item.percentage = (item.value / total) * 100;
                    });
                    
                    let sum = 0;
                    document.getElementById('output_token').innerHTML = "";  
                    
                    percentages.forEach(item => {
                        let jauge_width = item.percentage + 20;  
                        let jauge_value = item.percentage.toFixed(2);
                        sum += item.percentage; 
                        document.getElementById('output_token').innerHTML +=
                            `<div class="key_jauge">
                                <span id='key' onclick='addtext(this)' class='soft-btn'>${item.key}</span>
                                <div class="progress-container">
                                    <div class="progress">
                                        <div class="progress-bar" role="progressbar" 
                                            style="width:${jauge_width}%" aria-valuemin="0" aria-valuemax="100">
                                            ${jauge_value}%
                                        </div>
                                    </div>
                                </div>
                            </div>`;
                    });
                    
                  
      
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }, 1000); 
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

