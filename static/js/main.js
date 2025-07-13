let mediaRecorder;
let audioChunks = [];
let sessionId; // Set from index.html

document.addEventListener('DOMContentLoaded', () => {
    const pdfInput = document.getElementById('pdfInput');
    const textInput = document.getElementById('textInput');
    const recordButton = document.getElementById('recordButton');
    const stopRecordButton = document.getElementById('stopRecordButton');
    const submitButton = document.getElementById('submitButton');
    const resetButton = document.getElementById('resetButton');
    const summaryButton = document.getElementById('summaryButton');
    const audioOutput = document.getElementById('audioOutput');
    const micStatus = document.getElementById('micStatus');
    const resetStatus = document.getElementById('resetStatus');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const spinner = document.getElementById('spinner');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const summaryOutput = document.getElementById('summaryOutput');
    let isSummaryVisible = false;

    // Get sessionId from index.html
    sessionId = document.getElementById('sessionId').value;
    console.log('Initial session ID:', sessionId);

    const csrfToken = window.csrfToken;

    const updateSubmitButton = () => {
        const hasInput = textInput.value.trim() || audioChunks.length > 0 || pdfInput.files.length > 0;
        submitButton.disabled = !hasInput;
        submitButton.classList.toggle('gr-button-primary', hasInput);
        submitButton.classList.toggle('bg-gray-400', !hasInput);
        submitButton.classList.toggle('cursor-not-allowed', !hasInput);
    };

    updateSubmitButton();
    textInput.addEventListener('input', updateSubmitButton);
    pdfInput.addEventListener('change', updateSubmitButton);

    // Define supported MIME types
    const supportedMimeTypes = [
        { mime: 'audio/wav', ext: 'wav' },
        { mime: 'audio/mp3', ext: 'mp3' },
        { mime: 'audio/webm', ext: 'webm' },
        { mime: 'audio/ogg', ext: 'ogg' }
    ];

    // Find a supported MIME type
    let selectedMimeType = null;
    let fileExtension = 'wav';
    for (const type of supportedMimeTypes) {
        if (MediaRecorder.isTypeSupported(type.mime)) {
            selectedMimeType = type.mime;
            fileExtension = type.ext;
            break;
        }
    }

    // Initialize MediaRecorder
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            if (!selectedMimeType) {
                micStatus.textContent = 'Browser does not support any audio formats (WAV, MP3, WebM, OGG). Please use text input.';
                recordButton.disabled = true;
                recordButton.classList.add('bg-gray-400', 'cursor-not-allowed');
                recordButton.classList.remove('gr-button-primary');
                console.error('No supported audio MIME types');
                return;
            }
            micStatus.textContent = `Microphone access granted. Using ${fileExtension.toUpperCase()} format.`;
            mediaRecorder = new MediaRecorder(stream, { mimeType: selectedMimeType });
            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                    updateSubmitButton();
                }
            };
            mediaRecorder.onstop = () => {
                recordButton.classList.remove('hidden');
                stopRecordButton.classList.add('hidden');
                recordingIndicator.classList.add('hidden');
                micStatus.textContent = `Microphone access granted. ${fileExtension.toUpperCase()} format ready.`;
                console.log('Recording stopped');
            };
            mediaRecorder.onerror = (e) => {
                micStatus.textContent = `Recording error: ${e.error.message}. Please try again or use text input.`;
                recordButton.disabled = true;
                recordButton.classList.add('bg-gray-400', 'cursor-not-allowed');
                recordButton.classList.remove('gr-button-primary');
                console.error('MediaRecorder error:', e.error);
            };
        })
        .catch(err => {
            micStatus.textContent = `Microphone access error: ${err.message}. Use text input.`;
            recordButton.disabled = true;
            recordButton.classList.add('bg-gray-400', 'cursor-not-allowed');
            recordButton.classList.remove('gr-button-primary');
            console.error('Microphone error:', err);
        });

    recordButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'recording') {
            audioChunks = [];
            try {
                mediaRecorder.start();
                recordButton.classList.add('hidden');
                stopRecordButton.classList.remove('hidden');
                recordingIndicator.classList.remove('hidden');
                micStatus.textContent = 'Recording in progress...';
                console.log('Recording started');
            } catch (err) {
                micStatus.textContent = `Failed to start recording: ${err.message}.`;
                recordButton.disabled = true;
                recordButton.classList.add('bg-gray-400', 'cursor-not-allowed');
                recordButton.classList.remove('gr-button-primary');
                console.error('Recording start error:', err);
            }
        } else if (!mediaRecorder) {
            micStatus.textContent = 'Microphone not initialized. Please refresh and try again.';
            recordButton.disabled = true;
            recordButton.classList.add('bg-gray-400', 'cursor-not-allowed');
            recordButton.classList.remove('gr-button-primary');
        }
    });

    stopRecordButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            try {
                mediaRecorder.stop();
            } catch (err) {
                micStatus.textContent = `Failed to stop recording: ${err.message}.`;
                console.error('Recording stop error:', err);
            }
        }
    });

    submitButton.addEventListener('click', async () => {
        if (!textInput.value.trim() && audioChunks.length === 0 && pdfInput.files.length === 0) {
            loadingIndicator.querySelector('span').textContent = 'Failed: Please provide a PDF, text, or audio question.';
            return;
        }
        spinner.classList.remove('hidden');
        loadingIndicator.querySelector('span').textContent = 'Validating inputs...';
        const formData = new FormData();
        formData.append('session_id', sessionId);

        if (pdfInput.files.length > 0) {
            formData.append('pdf_file', pdfInput.files[0]);
            console.log('PDF file added to FormData');
        }
        if (audioChunks.length > 0) {
            const audioBlob = new Blob(audioChunks, { type: selectedMimeType });
            formData.append('audio_data', audioBlob, `question_${sessionId}.${fileExtension}`);
            console.log(`Audio data added to FormData as ${fileExtension}`);
        }
        if (textInput.value.trim()) {
            formData.append('text_input', textInput.value.trim());
            console.log('Text input added to FormData:', textInput.value.trim());
        }

        try {
            const response = await fetch('/ask/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken
                },
                body: formData
            });
            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);
            const text = await response.text();
            console.log('Raw response:', text);
            let result;
            try {
                result = JSON.parse(text);
            } catch (e) {
                console.error('JSON parse error:', e, 'Raw text:', text);
                throw new Error('Invalid JSON response from server');
            }
            if (result.audio_path) {
                audioOutput.src = result.audio_path;
                audioOutput.play();
                console.log('Audio output set to:', result.audio_path);
            }
            loadingIndicator.querySelector('span').textContent = result.status;
        } catch (err) {
            console.error('Submit error:', err);
            loadingIndicator.querySelector('span').textContent = `Failed: ${err.message}`;
        } finally {
            spinner.classList.add('hidden');
        }
    });

    summaryButton.addEventListener('click', async () => {
        isSummaryVisible = !isSummaryVisible;
        try {
            const response = await fetch('/summary/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ session_id: sessionId, is_visible: isSummaryVisible })
            });
            console.log('Summary response status:', response.status);
            const text = await response.text();
            console.log('Summary raw response:', text);
            let result;
            try {
                result = JSON.parse(text);
            } catch (e) {
                console.error('Summary JSON parse error:', e, 'Raw text:', text);
                throw new Error('Invalid JSON response from server');
            }
            summaryOutput.innerHTML = result.summary;
            isSummaryVisible = result.is_visible;
        } catch (err) {
            console.error('Summary error:', err);
            summaryOutput.innerHTML = `Failed: ${err.message}`;
        }
    });

    resetButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/reset/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ session_id: sessionId })
            });
            console.log('Reset response status:', response.status);
            const text = await response.text();
            console.log('Reset raw response:', text);
            let result;
            try {
                result = JSON.parse(text);
            } catch (e) {
                console.error('Reset JSON parse error:', e, 'Raw text:', text);
                throw new Error('Invalid JSON response from server');
            }
            resetStatus.textContent = result.status;
            if (result.session_id) {
                sessionId = result.session_id;
                document.getElementById('sessionId').value = sessionId;
                console.log('Updated session ID:', sessionId);
            }
            pdfInput.value = '';
            textInput.value = '';
            audioOutput.src = '';
            summaryOutput.innerHTML = '';
            audioChunks = [];
            isSummaryVisible = false;
            updateSubmitButton();
        } catch (err) {
            console.error('Reset error:', err);
            resetStatus.textContent = `Failed: ${err.message}`;
        }
    });
});