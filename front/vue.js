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


document.getElementById('input_token').addEventListener('input', function() {  
    const sentence = document.getElementById('input_token').value;
    const wordCount = sentence.split(/\s+/).filter(word => word.length > 0).length;

    if (wordCount > 3) {
        k = 5;
        getNextToken(sentence, k)
        .then(data => {
            document.getElementById('output_token').innerHTML = ''; 
            for (let key in data) {
                let value = data[key].toFixed(4);  
                document.getElementById('output_token').innerHTML += key + " : " + value + "<br>";
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        document.getElementById('output_token').innerHTML = '';
    }
});

