// DOM Elements
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section');
const analyzeBtn = document.getElementById('analyzeBtn');
const analysisOutput = document.getElementById('analysisOutput');
const typingContainer = document.getElementById('typingContainer');

// Store original button state for real-time detection
const originalBtnText = analyzeBtn ? analyzeBtn.querySelector('.btn-text').textContent : 'RUN DIAGNOSTICS';

// Real-time API check timeout
let checkTimeout = null;

// Flag to track if current input is a file upload
let isFileUpload = false;
let uploadedFileContent = null;

// --- Navigation Logic (SPA) ---
function navigateTo(targetId) {
    const landingSections = ['home', 'about', 'services', 'contact'];

    // Update Nav
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-target') === targetId) {
            link.classList.add('active');
        }
    });

    if (landingSections.includes(targetId)) {
        // Show Landing Page Wrapper
        document.getElementById('landing-page').classList.add('active-section');

        // Hide App Sections
        document.getElementById('scanner').classList.remove('active-section');
        document.getElementById('logs').classList.remove('active-section');

        // Scroll to specific section
        const element = document.getElementById(targetId);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }

    } else {
        // App View (Scanner / Logs)
        // Hide Landing Layout
        document.getElementById('landing-page').classList.remove('active-section');

        // Show Target App Section
        sections.forEach(section => {
            if (section.id === targetId) {
                section.classList.add('active-section');
            } else if (section.id !== 'landing-page' && !landingSections.includes(section.id)) {
                // Ensure other app sections are hidden
                section.classList.remove('active-section');
            }
        });
    }

    // Change Theme colors slightly based on section
    if (targetId === 'scanner' || targetId === 'logs') {
        document.documentElement.style.setProperty('--primary-color', '#00FF9D'); // Neon Mint
        document.documentElement.style.setProperty('--secondary-color', '#002920'); // Dark Mint
    } else {
        document.documentElement.style.setProperty('--primary-color', '#00FF9D');
        document.documentElement.style.setProperty('--secondary-color', '#008F6B'); // Normal Mint
    }
}

// Event Listeners for Nav
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = link.getAttribute('data-target');
        navigateTo(target);
    });
});

// --- Scanner Logic ---
const fileInput = document.getElementById('fileInput');
const uploadTrigger = document.getElementById('uploadTrigger');
const scannerInput = document.getElementById('scannerInput');

if (uploadTrigger && fileInput) {
    // Trigger hidden input
    uploadTrigger.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];

            // Set file upload flag
            isFileUpload = true;
            uploadedFileContent = null;

            // Show loading state
            uploadTrigger.classList.add('loading');
            scannerInput.value = `[LOADING FILE]...\n> FILE: ${file.name}\n> TYPE: ${file.type || 'unknown'}\n> SIZE: ${(file.size / 1024).toFixed(2)} KB\n\n[READING CONTENT...]`;

            // Reset button state for file uploads (don't block)
            updateButtonState('SAFE', '');

            // Read file content based on type
            try {
                if (file.type === 'application/pdf') {
                    // For PDF files, we'll send the file info to backend
                    scannerInput.value = `[FILE READY FOR ANALYSIS]\n> FILE: ${file.name}\n> TYPE: PDF Document\n> SIZE: ${(file.size / 1024).toFixed(2)} KB\n\n[Click RUN DIAGNOSTICS to analyze this PDF]`;
                    uploadedFileContent = {
                        type: 'pdf',
                        name: file.name,
                        size: file.size,
                        // We'll read as base64 for backend processing
                        data: await readFileAsBase64(file)
                    };
                } else if (file.type.startsWith('text/') ||
                           file.name.endsWith('.txt') ||
                           file.name.endsWith('.json') ||
                           file.name.endsWith('.csv') ||
                           file.name.endsWith('.md') ||
                           file.name.endsWith('.js') ||
                           file.name.endsWith('.py') ||
                           file.name.endsWith('.html') ||
                           file.name.endsWith('.css')) {
                    // Read text files directly
                    const content = await readFileAsText(file);
                    uploadedFileContent = {
                        type: 'text',
                        name: file.name,
                        content: content
                    };
                    // Show preview in textarea
                    const preview = content.substring(0, 500);
                    scannerInput.value = `[FILE CONTENT]\n> FILE: ${file.name}\n> TYPE: ${file.type || 'text'}\n> SIZE: ${(file.size / 1024).toFixed(2)} KB\n\n--- Content Preview ---\n${preview}${content.length > 500 ? '\n... (truncated)' : ''}`;

                    // Run detection on actual content
                    isFileUpload = false; // Allow detection on text content
                    const frontendResult = detect_light(content);
                    renderHighlights(scannerInput.value, []);
                    updateSecurityUI(frontendResult);
                } else {
                    // Other file types
                    scannerInput.value = `[FILE READY FOR ANALYSIS]\n> FILE: ${file.name}\n> TYPE: ${file.type || 'unknown'}\n> SIZE: ${(file.size / 1024).toFixed(2)} KB\n\n[Click RUN DIAGNOSTICS to analyze]`;
                    uploadedFileContent = {
                        type: 'binary',
                        name: file.name,
                        size: file.size
                    };
                }
            } catch (error) {
                console.error('File read error:', error);
                scannerInput.value = `[ERROR READING FILE]\n> FILE: ${file.name}\n> ERROR: ${error.message}\n\n[Please try a different file]`;
                uploadedFileContent = null;
            }

            uploadTrigger.classList.remove('loading');
        }
    });
}

/**
 * Read file as text
 */
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

/**
 * Read file as base64
 */
function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const base64 = e.target.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsDataURL(file);
    });
}

// ==================================================
// LIVE TEXTBOX HIGHLIGHTING (DYNAMIC + PROACTIVE)
// The textbox itself is the first security layer
// ==================================================

// Pattern definitions (same as backend, synced)
const PATTERNS = {
    HIGH: [
        // Instruction override
        /ignore\s+(the\s+)?(all\s+)?(previous|prior|earlier|initial|original|system|above)?\s*(instructions?|rules?|guidelines?|prompts?)/gi,
        /disregard\s+(the\s+)?(all\s+)?(previous|prior|earlier|initial|original|system|above)?\s*(instructions?|rules?|guidelines?|prompts?)/gi,
        /forget\s+(the\s+)?(all\s+)?(previous|prior|earlier|initial|original|system|above)?\s*(instructions?|rules?|guidelines?|prompts?)/gi,
        /remove\s+(the\s+)?(previous|all|system)?\s*(instructions?|rules?)/gi,
        /override\s+(the\s+)?(previous|system|all|above)?\s*(instructions?|rules?)/gi,
        // Safety bypass
        /\bunrestricted\b/gi,
        /\bunfiltered\b/gi,
        /developer\s+mode/gi,
        /jailbreak/gi,
        /bypass\s+(safety|filters?|restrictions?)/gi,
        /disable\s+(safety|filters?|moderation)/gi,
        /no\s+(filters?|restrictions?|rules?|limits?)/gi,
        // System extraction
        /(show|reveal|display|print)\s+(me\s+)?(your\s+)?(system\s+)?prompt/gi,

        // GROUP E: Malicious System Intent (NEW - semantic attacks)
        /attack\s+(the\s+)?(llm|model|system|ai)/gi,
        /exploit\s+(the\s+)?(llm|model|system|ai|vulnerability)/gi,
        /hack\s+(the\s+)?(llm|model|system|ai)/gi,
        /break\s+(the\s+)?(llm|model|system|ai|alignment)/gi,
        /(data|training)\s+poisoning/gi,
        /poison\s+(the\s+)?(data|training|model|llm)/gi,
        /corrupt\s+(the\s+)?(data|training|model|weights|llm)/gi,
        /manipulate\s+(the\s+)?(model|weights|training|llm)/gi,
        /sabotage\s+(the\s+)?(system|model|llm|ai)/gi,
        /adversarial\s+(attack|example|input|prompt)/gi,
        /malicious\s+(prompt|input|payload|code)/gi,
    ],
    MEDIUM: [
        // Roleplay
        /act\s+as\s+(a|an|my)?/gi,
        /you\s+are\s+(now\s+)?(a|an|my)?/gi,
        /behave\s+(like|as)/gi,
        /pretend\s+(to\s+be|you)/gi,
        /roleplay\s+as/gi,
        /assume\s+(the\s+)?(role|identity)/gi,
        /from\s+now\s+on\s+you/gi,
    ]
};


// Detection state
let detectionState = {
    status: 'SAFE',
    buttonEnabled: true,
    matches: [],
    reason: null
};

// Create highlight layer (mirror of textarea)
const highlightLayer = document.createElement('div');
highlightLayer.id = 'highlightLayer';
highlightLayer.className = 'highlight-layer';

// Insert highlight layer before textarea
if (scannerInput && scannerInput.parentNode) {
    scannerInput.parentNode.style.position = 'relative';
    scannerInput.parentNode.insertBefore(highlightLayer, scannerInput);
}

// Security alert element
const securityAlert = document.createElement('div');
securityAlert.id = 'securityAlert';
securityAlert.className = 'security-alert hidden';

if (analyzeBtn && analyzeBtn.parentNode) {
    analyzeBtn.parentNode.insertBefore(securityAlert, analyzeBtn);
}

/**
 * detect_light() - Frontend-only regex detection
 * Runs on EVERY keystroke, must be <1ms
 */
function detect_light(text) {
    if (!text || text.trim().length === 0) {
        return { status: 'SAFE', matches: [] };
    }

    const matches = [];
    let hasHigh = false;
    let hasMedium = false;

    // Check HIGH risk patterns
    for (const pattern of PATTERNS.HIGH) {
        pattern.lastIndex = 0; // Reset regex state
        let match;
        while ((match = pattern.exec(text)) !== null) {
            matches.push({
                phrase: match[0],
                start: match.index,
                end: match.index + match[0].length,
                severity: 'HIGH'
            });
            hasHigh = true;
        }
    }

    // Check MEDIUM risk patterns
    for (const pattern of PATTERNS.MEDIUM) {
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(text)) !== null) {
            matches.push({
                phrase: match[0],
                start: match.index,
                end: match.index + match[0].length,
                severity: 'MEDIUM'
            });
            hasMedium = true;
        }
    }

    // Sort by position, then by length (descending) to prioritize longer matches
    matches.sort((a, b) => {
        if (a.start !== b.start) return a.start - b.start;
        return b.end - a.end; // Longer match first
    });

    // Filter overlaps
    const uniqueMatches = [];
    let lastEnd = 0;

    for (const match of matches) {
        if (match.start >= lastEnd) {
            uniqueMatches.push(match);
            lastEnd = match.end;
        }
    }

    // Determine status
    let status = 'SAFE';
    if (hasHigh) status = 'BLOCK';
    else if (hasMedium) status = 'WARNING';

    return { status, matches: uniqueMatches };
}

/**
 * Render highlighted text in mirror layer
 */
function renderHighlights(text, matches) {
    if (!highlightLayer) return;

    if (matches.length === 0) {
        highlightLayer.innerHTML = escapeHtml(text) + '\n';
        return;
    }

    let html = '';
    let lastIndex = 0;

    for (const match of matches) {
        // Add text before match
        if (match.start > lastIndex) {
            html += escapeHtml(text.substring(lastIndex, match.start));
        }

        // Add highlighted match (ensure we don't go backwards)
        if (match.end > match.start) {
            const colorClass = match.severity === 'HIGH' ? 'hl-red' : 'hl-yellow';
            html += `<span class="${colorClass}">${escapeHtml(match.phrase)}</span>`;
            lastIndex = match.end;
        }
    }

    // Add remaining text
    if (lastIndex < text.length) {
        html += escapeHtml(text.substring(lastIndex));
    }

    highlightLayer.innerHTML = html + '\n';
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * Show input warning message below the textarea
 */
function showInputWarning(message, type = 'error') {
    let warningEl = document.getElementById('input-warning');
    if (!warningEl) {
        warningEl = document.createElement('div');
        warningEl.id = 'input-warning';
        warningEl.className = 'input-warning';
        if (scannerInput && scannerInput.parentElement) {
            scannerInput.parentElement.appendChild(warningEl);
        }
    }

    warningEl.className = `input-warning ${type}`;
    warningEl.textContent = `🚨 ${message}`;
    warningEl.style.display = 'block';
}

/**
 * Hide input warning message
 */
function hideInputWarning() {
    const warningEl = document.getElementById('input-warning');
    if (warningEl) {
        warningEl.style.display = 'none';
    }
}

/**
 * Update button state based on detection status (unified handler)
 */
function updateButtonState(status, reason) {
    if (!analyzeBtn) return;

    const btnText = analyzeBtn.querySelector('.btn-text');

    if (status === 'BLOCKED' || status === 'BLOCK') {
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('btn-blocked', 'disabled');
        analyzeBtn.classList.remove('btn-primary', 'btn-warning');
        btnText.textContent = '⛔ BLOCKED - THREAT DETECTED';
        analyzeBtn.title = reason || 'Prompt injection detected';

        // Show warning message
        showInputWarning(reason || 'High-risk injection pattern detected');
    } else if (status === 'WARNING') {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.add('btn-warning');
        analyzeBtn.classList.remove('btn-primary', 'btn-blocked', 'disabled');
        btnText.textContent = '⚠️ CAUTION - SUSPICIOUS';
        analyzeBtn.title = reason || 'Suspicious pattern detected';

        showInputWarning(reason || 'Suspicious intent detected', 'warning');
    } else {
        // SAFE
        analyzeBtn.disabled = false;
        analyzeBtn.classList.add('btn-primary');
        analyzeBtn.classList.remove('btn-blocked', 'btn-warning', 'disabled');
        btnText.textContent = originalBtnText;
        analyzeBtn.title = '';

        hideInputWarning();
    }
}

/**
 * Update UI based on detection state (modified to use unified handler)
 */
function updateSecurityUI(result) {
    detectionState = {
        status: result.status,
        buttonEnabled: result.status !== 'BLOCK',
        matches: result.matches,
        reason: result.status === 'BLOCK' ? 'high_risk_injection_detected' :
            result.status === 'WARNING' ? 'suspicious_intent_detected' : null
    };

    // Use unified button state handler
    updateButtonState(result.status, detectionState.reason);

    // Update security alert (Inline only logic: Hide banner for ALL states)
    if (securityAlert) {
        securityAlert.className = 'security-alert hidden';
        securityAlert.innerHTML = '';
    }
}

/**
 * Real-time backend API check for prompt injection
 */
async function checkForInjectionAPI(text) {
    // Skip API check for file metadata
    if (isFileUpload || isFileMetadata(text)) {
        return null;
    }

    if (!text || text.trim().length < 3) {
        return null;
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/api/realtime-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Realtime API check failed:', error);
    }

    return null; // Return null on error, fall back to frontend detection
}

/**
 * Check if text is file upload metadata (should skip detection)
 */
function isFileMetadata(text) {
    const metadataPatterns = [
        /^\[LOADING FILE/,
        /^\[FILE READY FOR ANALYSIS\]/,
        /^\[FILE CONTENT\]/,
        /^\[ERROR READING FILE\]/,
        /^\[READING CONTENT\.\.\.\]/,
        /^> FILE:/m,
    ];
    return metadataPatterns.some(pattern => pattern.test(text));
}

/**
 * Sync textarea scroll with highlight layer
 */
function syncScroll() {
    if (highlightLayer && scannerInput) {
        highlightLayer.scrollTop = scannerInput.scrollTop;
        highlightLayer.scrollLeft = scannerInput.scrollLeft;
    }
}

// Live detection on EVERY keystroke (combined frontend + backend)
if (scannerInput) {
    scannerInput.addEventListener('input', (e) => {
        const text = e.target.value;

        // Reset file upload flag when user types manually
        if (!isFileMetadata(text)) {
            isFileUpload = false;
            uploadedFileContent = null;
        }

        // Skip detection for file metadata
        if (isFileUpload || isFileMetadata(text)) {
            renderHighlights(text, []);
            updateButtonState('SAFE', '');
            return;
        }

        // 1. INSTANT: Frontend regex detection (no delay)
        const frontendResult = detect_light(text);

        // Render highlights in mirror layer
        renderHighlights(text, frontendResult.matches);

        // Update button + alert based on frontend detection immediately
        updateSecurityUI(frontendResult);

        // 2. DEBOUNCED: Backend API check (300ms delay)
        if (checkTimeout) {
            clearTimeout(checkTimeout);
        }

        checkTimeout = setTimeout(async () => {
            const apiResult = await checkForInjectionAPI(text);

            if (apiResult) {
                // Backend result takes precedence if it's more severe
                const frontendSeverity = frontendResult.status === 'BLOCK' ? 2 : frontendResult.status === 'WARNING' ? 1 : 0;
                const apiSeverity = apiResult.status === 'BLOCKED' ? 2 : apiResult.status === 'WARNING' ? 1 : 0;

                if (apiSeverity >= frontendSeverity) {
                    updateButtonState(apiResult.status, apiResult.reason);
                }
            }
        }, 300);
    });

    scannerInput.addEventListener('scroll', syncScroll);
}

analyzeBtn.addEventListener('click', async () => {
    const input = document.getElementById('scannerInput').value;
    if (!input.trim()) return;

    // 1. UI Loading State
    analyzeBtn.classList.add('loading');
    analyzeBtn.querySelector('.btn-text').innerText = 'SCANNING...';

    // Clear previous output and show loading message
    analysisOutput.classList.remove('hidden');
    typingContainer.innerHTML = '> Establishing link to Trinetra Core...\n> Awaiting agent response...';

    // Determine what to send to backend
    let payloadToSend = input;
    let logMessage = input.substring(0, 40) + (input.length > 40 ? '...' : '');

    // If we have uploaded file content, prepare the payload
    if (uploadedFileContent) {
        if (uploadedFileContent.type === 'text') {
            payloadToSend = `Analyze this file content:\n\nFilename: ${uploadedFileContent.name}\n\nContent:\n${uploadedFileContent.content}`;
            logMessage = `File: ${uploadedFileContent.name}`;
        } else if (uploadedFileContent.type === 'pdf') {
            payloadToSend = `Analyze this PDF file:\n\nFilename: ${uploadedFileContent.name}\nSize: ${(uploadedFileContent.size / 1024).toFixed(2)} KB\n\n[PDF content analysis requested]`;
            logMessage = `PDF: ${uploadedFileContent.name}`;
        } else {
            payloadToSend = `Analyze this file:\n\nFilename: ${uploadedFileContent.name}\nSize: ${(uploadedFileContent.size / 1024).toFixed(2)} KB`;
            logMessage = `File: ${uploadedFileContent.name}`;
        }
    }

    // Log the user scan
    addToLogs('USER_SCAN', logMessage, 'INFO');

    try {
        // 2. Connect to Python Backend
        const response = await fetch('http://127.0.0.1:8000/scan', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ payload: payloadToSend })
        });

        const data = await response.json();

        // 3. Remove Loading State
        analyzeBtn.classList.remove('loading');
        analyzeBtn.querySelector('.btn-text').innerText = 'RUN DIAGNOSTICS';
        typingContainer.innerHTML = ''; // Clear the loading text

        // 4. Display Result based on 2-Phase Security Flow
        if (data.status === 'BLOCKED') {
            // 🔴 BLOCKED: Show alert, risk level, and reason
            const formattedOutput =
                `> [TRINETRA SECURITY ALERT]\n` +
                `> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n` +
                `> ${data.alert || '🚫 Prompt Injection Detected'}\n` +
                `> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n` +
                `> [DECISION]: ${data.decision || 'BLOCK'}\n` +
                `> [STATUS]: BLOCKED\n` +
                `> [RISK LEVEL]: ${data.risk_level || 'HIGH'}\n` +
                `> [REASON]: ${data.reason || 'Malicious pattern detected'}\n` +
                (data.matched_patterns && data.matched_patterns.length > 0
                    ? `> [PATTERNS]: ${data.matched_patterns.slice(0, 3).join(', ')}\n`
                    : '') +
                `> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n` +
                `> [ACTION]: Input rejected. No LLM execution.`;

            typeWriter(formattedOutput, 0);
            addToLogs('THREAT_BLOCKED', data.reason || 'Prompt Injection', 'HIGH');

        } else if (data.decision === 'ALLOW_WITH_WARNING') {
            // 🟡 WARNING → ALLOW: Show warning banner + explanation
            const explanation = data.explanation || 'No analysis available';
            const formattedOutput =
                `> [TRINETRA SECURITY NOTICE]\n` +
                `> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n` +
                `> ⚠️ Suspicious Intent Detected\n` +
                `------------------------------------\n` +
                explanation;

            typeWriter(formattedOutput, 0);
            addToLogs('SYS_WARNING', 'Allowed with Warning', 'MEDIUM');

        } else if (data.status === 'ALLOWED') {
            // 🟢 SAFE: Show clean analysis
            const explanation = data.explanation || 'No analysis available';
            const formattedOutput =
                `> [TRINETRA SECURITY NOTICE]\n` +
                `> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n` +
                `> [ANALYSIS]:\n` +
                `------------------------------------\n` +
                explanation;

            typeWriter(formattedOutput, 0);
            addToLogs('SYS_RESPONSE', 'Analysis Received', 'LOW');

        } else {
            // Legacy fallback
            if (data.analysis) {
                typeWriter(`> [TRINETRA AGENT REPORT]\n> [STATUS]: COMPLETE\n------------------------------------\n${data.analysis}`, 0);
            } else {
                typeWriter(`> [ERROR]: Unexpected response format\n> ${JSON.stringify(data)}`, 0);
            }
            addToLogs('SYS_RESPONSE', 'Legacy Response', 'LOW');
        }

    } catch (error) {
        // Handle connection errors (e.g., Python script not running)
        console.error('Connection failed:', error);
        analyzeBtn.classList.remove('loading');
        analyzeBtn.querySelector('.btn-text').innerText = 'RETRY';
        typingContainer.innerHTML = '';
        typeWriter(`> [CRITICAL ERROR]\n> Connection to Trinetra Core refused.\n> Ensure backend port 8000 is active.\n> Run: python main.py`, 0);
        addToLogs('SYS_ERROR', 'Backend Connection Failed', 'HIGH');
    }

    // Reset file upload state after scan
    isFileUpload = false;
    uploadedFileContent = null;

    // Refresh metrics and URL data after scan
    loadMetrics();
    loadUrlResults();
});

// --- SECURITY DASHBOARD ---
const urlList = document.getElementById('urlList');

// URL classification state
let urlState = {
    safe: [],
    threats: [],
    activeTab: 'safe'
};

// Fetch metrics from backend (single source of truth)
async function loadMetrics() {
    try {
        const res = await fetch('http://127.0.0.1:8000/metrics');
        const data = await res.json();

        const scansEl = document.getElementById('stats-scans');
        if (scansEl) scansEl.innerText = data.total_scans.toLocaleString();
    } catch (e) {
        console.log('Metrics unavailable:', e);
    }
}

// Load URL classifications from backend
async function loadUrlResults() {
    try {
        const res = await fetch('http://127.0.0.1:8000/scan/urls');
        const data = await res.json();

        urlState.safe = data.safe || [];
        urlState.threats = data.threats || [];

        // Update counters
        const safeEl = document.getElementById('stats-safe');
        const threatsEl = document.getElementById('stats-threats');

        if (safeEl) safeEl.innerText = data.safe_count.toLocaleString();
        if (threatsEl) threatsEl.innerText = data.threat_count.toLocaleString();

        renderCurrentTab();
    } catch (e) {
        console.log('URL results unavailable:', e);
    }
}

// Tab switching
function switchTab(tab) {
    urlState.activeTab = tab;

    const tabSafe = document.getElementById('tabSafe');
    const tabThreats = document.getElementById('tabThreats');

    if (tabSafe) tabSafe.classList.toggle('active', tab === 'safe');
    if (tabThreats) tabThreats.classList.toggle('active', tab === 'threats');

    renderCurrentTab();
}

function renderCurrentTab() {
    if (urlState.activeTab === 'safe') {
        renderSafe();
    } else {
        renderThreats();
    }
}

// Render safe URLs
function renderSafe() {
    if (!urlList) return;
    urlList.innerHTML = '';

    if (urlState.safe.length === 0) {
        urlList.innerHTML = '<div class="empty-state">No safe URLs yet. Run a scan that uses external sources.</div>';
        return;
    }

    urlState.safe.forEach(item => {
        const div = document.createElement('div');
        div.className = 'url-card safe';
        div.innerHTML = `
            <span class="url-icon">✅</span>
            <span class="url-text">${item.url}</span>
            <span class="url-reason">${item.reason || 'Trusted source'}</span>
            <button class="open-btn" onclick="openSite('${item.url}')">Open ↗</button>
        `;
        urlList.appendChild(div);
    });
}

// Render threat URLs
function renderThreats() {
    if (!urlList) return;
    urlList.innerHTML = '';

    if (urlState.threats.length === 0) {
        urlList.innerHTML = '<div class="empty-state">No threats detected yet.</div>';
        return;
    }

    urlState.threats.forEach(item => {
        const div = document.createElement('div');
        div.className = 'url-card threat';
        div.innerHTML = `
            <span class="url-icon">❌</span>
            <span class="url-text">${item.url}</span>
            <span class="url-reason">${item.reason || 'Low credibility'}</span>
        `;
        urlList.appendChild(div);
    });
}

// Open safe site only
function openSite(url) {
    window.open(url, '_blank', 'noopener,noreferrer');
}

// Legacy logging for scan events (minimal)
function addToLogs(type, details, severity) {
    // No longer rendering to DOM - dashboard is URL-focused now
    console.log(`[${type}] ${details} (${severity})`);
}

function generateAnalysis(input) {
    // Simple heuristic for demo purposes
    const isSuspicious = input.toLowerCase().includes('drop') || input.toLowerCase().includes('delete') || input.toLowerCase().includes('ignore previous');
    const pplScore = (Math.random() * (50 - 5) + 5).toFixed(2);

    let logs = `> [INITIATING TRINETRA PIPELINE] on input...\n` +
        `> [PRE-PROCESS]: Tokenizing & Paraphrasing (T5 Model)... DONE\n`;

    if (isSuspicious) {
        logs += `> [DETECTION]: Calculating Perplexity (PPL)... SCORE: ${pplScore} (HIGH)\n` +
            `> [ALERT]: Anomalous pattern detected (Prompt Injection Signature)\n` +
            `> [DECISION]: ★ FLAGGING AS SUSPICIOUS ★\n` +
            `> [ACTION]: Rerouting to **SANDBOX** for isolation.\n` +
            `\n[RESULT]: BLOCKED. Input treated as raw data. No execution allowed.`;
    } else {
        logs += `> [DETECTION]: Calculating Perplexity (PPL)... SCORE: ${pplScore} (NORMAL)\n` +
            `> [VALIDATION]: Keyword Scan... CLEAN\n` +
            `> [DECISION]: Input Classification -> SAFE\n` +
            `> [ACTION]: Forwarding to LLM Context Window.\n` +
            `\n[RESULT]: VERIFIED. Payload delivered to model.`;
    }

    return logs;
}

function typeWriter(text, i) {
    if (i < text.length) {
        typingContainer.innerHTML += text.charAt(i);
        // Scroll to bottom
        document.querySelector('.output-body').scrollTop = document.querySelector('.output-body').scrollHeight;
        setTimeout(() => typeWriter(text, i + 1), 30);
    }
}

// --- 3D Wireframe Grid Background & Parallax ---
const canvas = document.getElementById('bgCanvas');
const ctx = canvas.getContext('2d');

let width, height;
function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
}
window.addEventListener('resize', resize);
resize();

// Grid Parameters
const gridSize = 40;
const speed = 0.5;
let offset = 0;
let parallaxY = 0;

// Parallax Scroll Effect
window.addEventListener('scroll', () => {
    parallaxY = window.scrollY;
});

function drawGrid() {
    // Clear
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#00FF9D'; // Neon Mint / Cyber Green
    ctx.lineWidth = 1;

    // Horizon is fixed relative to screen, but elements move with parallax
    const horizon = height * 0.4;

    ctx.save();
    // Parallax: Move the entire grid UP/DOWN based on scroll
    // but keep it subtle.
    ctx.translate(0, -parallaxY * 0.1);

    // Mute bottom part fade
    const gradient = ctx.createLinearGradient(0, horizon, 0, height);
    gradient.addColorStop(0, 'rgba(0, 255, 157, 0)');
    gradient.addColorStop(0.2, 'rgba(0, 255, 157, 0.1)');
    gradient.addColorStop(1, 'rgba(0, 255, 157, 0.4)'); // Stronger at bottom

    // Vertical Lines (Simulating infinite depth)
    for (let x = -width; x < width * 2; x += gridSize) {
        ctx.beginPath();
        // Lines converge to a vanishing point at horizon
        ctx.moveTo(x + (width / 2 - x) * 0.6, horizon);
        ctx.lineTo(x, height * 2);
        ctx.strokeStyle = gradient;
        ctx.stroke();
    }

    // Horizontal Lines (Moving forward)
    const time = Date.now() * 0.002;

    // Draw "floor" lines
    for (let z = 0; z < 40; z++) {
        // Perspective logic
        const p = (z + (offset / gridSize)) / 25;

        if (p > 0) {
            const y = height - (Math.pow(p, 2) * (height - horizon));

            if (y > horizon && y < height * 1.5) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);

                // PULSE ANIMATION logic for "Alive" feel
                const pulse = (Math.sin(time * 0.5) + 1) / 2; // 0 to 1
                const baseAlpha = (p < 0.8) ? p * 0.6 : (1 - p);
                const finalAlpha = baseAlpha * (0.5 + (pulse * 0.5));

                ctx.strokeStyle = `rgba(0, 255, 65, ${finalAlpha})`;
                ctx.stroke();
            }
        }
    }

    ctx.restore();

    offset += speed;
    if (offset > gridSize) offset = 0;

    requestAnimationFrame(drawGrid);
}

drawGrid();

// --- Micro-Interactions: Typing Reveal ---
const observerOptions = {
    threshold: 0.1
};

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const el = entry.target;
            if (!el.hasAttribute('data-typed')) {
                el.style.visibility = 'visible';
                el.classList.add('typing-active');
                el.setAttribute('data-typed', 'true');

                // Remove cursor after typing finishes
                el.addEventListener('animationend', (e) => {
                    if (e.animationName === 'typing') {
                        el.style.borderRight = 'none';
                    }
                });
            }
        }
    });
}, observerOptions);

document.querySelectorAll('.type-reveal').forEach(el => {
    el.style.visibility = 'hidden';
    revealObserver.observe(el);
});

// Inject CSS for typing animation
if (!document.getElementById('typing-style')) {
    const styleSheet = document.createElement("style");
    styleSheet.id = 'typing-style';
    styleSheet.innerText = `
        .typing-active {
            overflow: hidden; 
            border-right: .15em solid var(--primary-color); 
            white-space: nowrap;
            /* Adjust step count to fit text roughly */ 
            animation: typing 2.5s steps(30, end), blink-caret .75s step-end infinite;
        }
        @keyframes typing { from { width: 0 } to { width: 100% } }
        @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: var(--primary-color); } }
    `;
    document.head.appendChild(styleSheet);
}

// --- Static Cursor Block on Headers ---
function addBlinkingCursor() {
    const headers = document.querySelectorAll('h2');
    headers.forEach(header => {
        // Target specific headers mentioned: "Establish Connection" 
        if (header.innerText.includes('CONNECTION')) {
            // Check if already added
            if (!header.querySelector('.cursor-block')) {
                const cursorSpan = document.createElement('span');
                cursorSpan.className = 'cursor-block';
                header.appendChild(cursorSpan);
            }
        }
    });
}
// Run once on load
addBlinkingCursor();

// --- Init ---
// Check URL hash on load
const hash = window.location.hash.substring(1);
if (hash) {
    navigateTo(hash);
} else {
    navigateTo('home');
}

// Load real metrics and URL data from backend on startup
loadMetrics();
loadUrlResults();
