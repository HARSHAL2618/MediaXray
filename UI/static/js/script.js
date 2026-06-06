function showLoader(id){
    document.getElementById(id).style.display = "block";
}

function hideLoader(id){
    document.getElementById(id).style.display = "none";
}

function getResultClass(label){
    return label.toLowerCase().includes("fake") ||
           label.toLowerCase().includes("ai")
           ? "fake"
           : "real";
}

/* IMAGE */
async function detectImage(){

    const fileInput = document.getElementById("imageInput");
    const resultBox = document.getElementById("imageResult");

    if(!fileInput.files.length){
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    showLoader("imageLoader");

    const response = await fetch("/predict/image",{
        method:"POST",
        body:formData
    });

    const data = await response.json();

    hideLoader("imageLoader");

    resultBox.innerHTML = `
        <div class="result-card">
            <div class="result-label ${getResultClass(data.label)}">
                ${data.label}
            </div>

            <div class="confidence">
                Confidence: ${data.confidence}%
            </div>

            <div class="confidence">
                AI Tool: ${data.tool}
            </div>

            <img
                src="data:image/jpeg;base64,${data.original}"
                class="preview-img"
            >

            <img
                src="data:image/jpeg;base64,${data.gradcam}"
                class="preview-img"
            >
        </div>
    `;
}

/* VIDEO */
async function detectVideo(){

    const fileInput = document.getElementById("videoInput");
    const resultBox = document.getElementById("videoResult");

    if(!fileInput.files.length){
        alert("Please upload a video.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    showLoader("videoLoader");

    const response = await fetch("/predict/video",{
        method:"POST",
        body:formData
    });

    const data = await response.json();

    hideLoader("videoLoader");

    resultBox.innerHTML = `
        <div class="result-card">

            <div class="result-label ${getResultClass(data.label)}">
                ${data.label}
            </div>

            <div class="confidence">
                Confidence: ${data.confidence}%
            </div>

            <div class="confidence">
                Frames Analyzed: ${data.frames_analyzed}
            </div>

            <div class="confidence">
                Fake Frames: ${data.fake_frames}
            </div>

            <div class="confidence">
                Real Frames: ${data.real_frames}
            </div>

        </div>
    `;
}

/* AUDIO */
async function detectAudio(){

    const fileInput = document.getElementById("audioInput");
    const resultBox = document.getElementById("audioResult");

    if(!fileInput.files.length){
        alert("Please upload audio.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    showLoader("audioLoader");

    const response = await fetch("/predict/audio",{
        method:"POST",
        body:formData
    });

    const data = await response.json();

    hideLoader("audioLoader");

    resultBox.innerHTML = `
        <div class="result-card">

            <div class="result-label ${getResultClass(data.label)}">
                ${data.label}
            </div>

            <div class="confidence">
                Confidence: ${data.confidence}%
            </div>

        </div>
    `;
}

/* TEXT */
async function detectText(){

    const text = document.getElementById("textInput").value;
    const resultBox = document.getElementById("textResult");

    if(text.trim() === ""){
        alert("Please enter text.");
        return;
    }

    showLoader("textLoader");

    const response = await fetch("/predict/text",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({text:text})
    });

    const data = await response.json();

    hideLoader("textLoader");

    resultBox.innerHTML = `
        <div class="result-card">

            <div class="result-label ${getResultClass(data.label)}">
                ${data.label}
            </div>

            <div class="confidence">
                Confidence: ${data.confidence}%
            </div>

            <div class="confidence">
                Human Probability: ${data.human_prob}%
            </div>

            <div class="confidence">
                AI Probability: ${data.ai_prob}%
            </div>

        </div>
    `;
}