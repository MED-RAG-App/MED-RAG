const API = "http://127.0.0.1:8000";
const PATIENT_ID = "demo_patient_001";   // must match ^[A-Za-z0-9_-]{1,64}$
const MAX_FILE_MB = 15;                  // keep equal to MAX_FILE_SIZE_MB in backend/.env
const STORAGE_KEY = "medrag_chat_history";

let currentDocuments = [];
let chatHistory = [];
let currentChatId = null;

let lastAnswer = "";
let lastQuestion = "";

// Sources are kept here so buttons can reference them by index
// (avoids putting JSON with quotes inside inline onclick attributes).
let sourceRegistry = [];

const suggestionPool = [
    "What is the patient's age?",
    "What are the symptoms?",
    "What are the vital signs?",
    "List the medications",
    "Are there any allergies?",
    "What is the diagnosis?",
    "Summarize the documents",
    "मरीज की उम्र कितनी है?",
    "मरीज कौन सी दवाइयाँ ले रहा है?",
    "ರೋಗಿಯ ವಯಸ್ಸು ಎಷ್ಟು?",
    "ರೋಗಿಗೆ ಯಾವ ಔಷಧಿಗಳನ್ನು ನೀಡಲಾಗಿದೆ?"
];

document.addEventListener("DOMContentLoaded", () => {
    loadHistory();
    updateDocuments();
    document.getElementById("question").focus();
});


/* =========================
   HELPERS
========================= */

function escapeHTML(value) {
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
}

function formatAnswer(text) {
    return escapeHTML(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}

async function readJson(response) {
    try {
        return await response.json();
    } catch {
        return {};
    }
}

function errorText(data, fallback) {
    if (!data || data.detail === undefined) return fallback;
    if (typeof data.detail === "string") return data.detail;
    try {
        return data.detail.map(d => d.msg).join("; ") || fallback;
    } catch {
        return fallback;
    }
}

function getEl(id) {
    return document.getElementById(id);
}


/* =========================
   FILE UPLOAD
========================= */

function openFilePicker() {
    getEl("fileInput").click();
}

async function handleFiles(fileList) {

    const files = Array.from(fileList || []);
    getEl("fileInput").value = "";       // allow re-selecting the same file

    if (!files.length) return;

    const validFiles = files.filter(file => {

        const isPdf =
            file.type === "application/pdf" ||
            file.name.toLowerCase().endsWith(".pdf");

        if (!isPdf) {
            alert(`${file.name}: only PDF files are supported.`);
            return false;
        }

        if (file.size > MAX_FILE_MB * 1024 * 1024) {
            alert(`${file.name} is larger than ${MAX_FILE_MB} MB.`);
            return false;
        }

        return true;
    });

    if (!validFiles.length) return;

    showUploading(validFiles);

    const uploaded = [];
    const failed = [];

    for (const file of validFiles) {
        try {
            await uploadSingleFile(file);
            uploaded.push(file);
        } catch (error) {
            console.error(error);
            failed.push(`${file.name}: ${error.message || "upload failed"}`);
        }
    }

    updateDocuments();

    if (failed.length) {
        alert("Some files could not be added:\n\n" + failed.join("\n"));
    }

    if (!uploaded.length) return;

    getEl("welcome").style.display = "none";

    if (!currentChatId) createCurrentChat();

    saveCurrentChat();

    addSystemMessage(
        uploaded.length === 1
            ? `${uploaded[0].name} is ready.`
            : `${uploaded.length} documents are ready.`
    );

    getEl("question").focus();
}

async function uploadSingleFile(file) {

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", PATIENT_ID);      // required by the backend

    let response;

    try {
        response = await fetch(`${API}/upload-pdf`, {
            method: "POST",
            body: formData
        });
    } catch {
        throw new Error("Cannot reach the MED-RAG backend. Is it running on " + API + "?");
    }

    const data = await readJson(response);

    if (!response.ok) {
        throw new Error(errorText(data, `Upload failed (${response.status})`));
    }

    const documentData = {
        name: file.name,
        type: "application/pdf",
        size: file.size,
        pages: data.pages || 1,
        ocr: false,
        documentId: data.document_id || "",
        url: URL.createObjectURL(file)
    };

    const existingIndex = currentDocuments.findIndex(doc => doc.name === file.name);

    if (existingIndex >= 0) {
        if (currentDocuments[existingIndex].url) {
            URL.revokeObjectURL(currentDocuments[existingIndex].url);
        }
        currentDocuments[existingIndex] = documentData;
    } else {
        currentDocuments.push(documentData);
    }
}


/* =========================
   DOCUMENT LIST
========================= */

function showUploading(files) {

    getEl("documents").innerHTML = files.map(file => `
        <div class="document-item">
            <div class="file-icon">...</div>
            <div class="document-name">
                <b>${escapeHTML(file.name)}</b>
                <span>Reading and indexing document...</span>
            </div>
        </div>
    `).join("");
}

function updateDocuments() {

    const box = getEl("documents");

    getEl("documentCount").textContent = currentDocuments.length;

    if (!currentDocuments.length) {
        box.innerHTML = "";
        return;
    }

    box.innerHTML = currentDocuments.map((doc, index) => `
        <div class="document-item">
            <div class="file-icon">PDF</div>
            <div class="document-name">
                <b title="${escapeHTML(doc.name)}">${escapeHTML(doc.name)}</b>
                <span>${doc.pages || 1} page${(doc.pages || 1) === 1 ? "" : "s"}</span>
            </div>
            <button class="doc-remove" onclick="removeDocument(${index})" title="Remove">×</button>
        </div>
    `).join("");
}

function removeDocument(index) {

    const doc = currentDocuments[index];
    if (!doc) return;

    if (doc.url) URL.revokeObjectURL(doc.url);

    currentDocuments.splice(index, 1);

    updateDocuments();

    if (!currentDocuments.length) {
        getEl("welcome").style.display = "block";
    }

    saveCurrentChat();
}

function releaseDocumentUrls() {
    currentDocuments.forEach(doc => {
        if (doc.url) URL.revokeObjectURL(doc.url);
    });
}

function documentsSnapshot() {
    return currentDocuments.map(doc => ({
        name: doc.name,
        type: doc.type,
        size: doc.size,
        pages: doc.pages,
        ocr: doc.ocr,
        documentId: doc.documentId || ""
    }));
}


/* =========================
   CHAT CREATION / SAVING
========================= */

function createCurrentChat() {

    if (!currentDocuments.length) return;

    currentChatId =
        "chat_" + Date.now() + "_" + Math.random().toString(36).slice(2);

    chatHistory.unshift({
        id: currentChatId,
        documents: documentsSnapshot(),
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    });

    saveHistoryToStorage();
    renderHistory();
}

function findCurrentChat() {
    return chatHistory.find(item => item.id === currentChatId);
}

function saveCurrentChat() {

    const chat = findCurrentChat();
    if (!chat) return;

    chat.documents = documentsSnapshot();
    chat.updatedAt = new Date().toISOString();

    saveHistoryToStorage();
    renderHistory();
}

function saveMessage(role, text, sources = []) {

    const chat = findCurrentChat();
    if (!chat) return;

    chat.messages.push({
        role: role,
        text: text,
        sources: sources,
        time: new Date().toISOString()
    });

    chat.updatedAt = new Date().toISOString();

    saveHistoryToStorage();
    renderHistory();
}


/* =========================
   SEND QUESTION
========================= */

async function sendQuestion() {

    const input = getEl("question");
    const question = input.value.trim();

    if (!question) return;

    if (!currentDocuments.length) {
        alert("Please upload at least one medical document first.");
        return;
    }

    if (!currentChatId) createCurrentChat();

    lastQuestion = question;

    addUserMessage(question);
    saveMessage("user", question);

    input.value = "";
    autoResize(input);

    getEl("suggestions").innerHTML = "";

    const loading = addLoadingMessage();
    scrollChat();

    try {

        const documentIds = currentDocuments
            .map(doc => doc.documentId)
            .filter(Boolean);

        const response = await fetch(`${API}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                patient_id: PATIENT_ID,        // required by the backend
                document_ids: documentIds      // limit search to the listed documents
            })
        });

        const data = await readJson(response);

        if (!response.ok) {
            throw new Error(errorText(data, `Question failed (${response.status})`));
        }

        loading.remove();

        lastAnswer = data.answer || "";

        const sources = data.sources || [];
        const text = data.answer || "No answer was generated.";

        addAIMessage(text, sources);
        saveMessage("assistant", text, sources);
        saveCurrentChat();

    } catch (error) {

        loading.remove();
        console.error(error);

        const isNetwork = error instanceof TypeError;

        addAIMessage(
            isNetwork
                ? "I couldn't reach the MED-RAG backend. Please check that it is running."
                : `I couldn't get an answer: ${error.message}`,
            []
        );
    }
}


/* =========================
   MESSAGES
========================= */

function addUserMessage(text) {

    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `<div class="user-bubble">${escapeHTML(text)}</div>`;

    getEl("messages").appendChild(row);
    scrollToMessage(row);
}

function addLoadingMessage() {

    const row = document.createElement("div");
    row.className = "ai-row";
    row.innerHTML = `
        <div class="ai-avatar">✦</div>
        <div class="ai-card"><p>Reading your documents...</p></div>
    `;

    getEl("messages").appendChild(row);
    scrollToMessage(row);

    return row;
}

function addAIMessage(answer, sources) {

    const row = document.createElement("div");
    row.className = "ai-row";

    const uniqueSources = [];
    const seen = new Set();

    (sources || []).forEach(source => {
        const key = `${source.filename}|${source.page}`;
        if (!seen.has(key)) {
            seen.add(key);
            uniqueSources.push(source);
        }
    });

    const buttons = uniqueSources.map(source => {
        const idx = sourceRegistry.push(source) - 1;
        return `
            <button type="button" class="source-btn" onclick="showSourceByIndex(${idx})">
                ${escapeHTML(source.filename)} • Page ${source.page || 1}
            </button>`;
    }).join("");

    const sourceHTML = uniqueSources.length
        ? `<div class="sources"><span class="sources-label">Sources:</span>${buttons}</div>`
        : "";

    row.innerHTML = `
        <div class="ai-avatar">✦</div>
        <div class="ai-card">
            <p>${formatAnswer(answer)}</p>
            ${sourceHTML}
        </div>
    `;

    getEl("messages").appendChild(row);
    scrollToMessage(row);
}

function addSystemMessage(text) {

    const row = document.createElement("div");
    row.className = "ai-row";
    row.innerHTML = `
        <div class="ai-avatar">✦</div>
        <div class="ai-card"><p>${escapeHTML(text)}</p></div>
    `;

    getEl("messages").appendChild(row);
    scrollToMessage(row);
}


/* =========================
   SCROLL
========================= */

function scrollChat() {

    const chat = getEl("chatScroll");

    requestAnimationFrame(() => {
        chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
    });
}

function scrollToMessage(element) {

    requestAnimationFrame(() => {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}


/* =========================
   RECENT CHAT HISTORY
========================= */

function saveHistoryToStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory));
    } catch (error) {
        console.warn("Could not save chat history", error);
    }
}

function loadHistory() {

    try {
        chatHistory = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch {
        chatHistory = [];
    }

    renderHistory();
}

function renderHistory() {

    const list = getEl("chatList");

    if (!chatHistory.length) {
        list.innerHTML = "";
        return;
    }

    const sorted = [...chatHistory].sort(
        (a, b) => new Date(b.updatedAt) - new Date(a.updatedAt)
    );

    list.innerHTML = sorted.map(chat => {

        const count = chat.messages ? chat.messages.length : 0;
        const questionCount = Math.floor(count / 2);
        const names = (chat.documents || []).map(doc => doc.name);

        const title = names.length === 1
            ? names[0]
            : `${names.length} Medical Documents`;

        return `
            <div class="history-chat ${chat.id === currentChatId ? "active" : ""}">

                <button type="button" class="history-open" onclick="openChat('${chat.id}')">
                    <div class="history-title">
                        <span class="history-icon">${names.length > 1 ? "📚" : "📄"}</span>
                        <span class="history-name" title="${escapeHTML(title)}">${escapeHTML(title)}</span>
                    </div>
                    <div class="history-meta">
                        ${questionCount} question${questionCount === 1 ? "" : "s"}
                    </div>
                </button>

                <button type="button" class="history-delete"
                        onclick="deleteChat(event, '${chat.id}')" title="Delete chat">⋮</button>

            </div>
        `;
    }).join("");
}

function openChat(chatId) {

    const chat = chatHistory.find(item => item.id === chatId);
    if (!chat) return;

    currentChatId = chat.id;

    releaseDocumentUrls();
    currentDocuments = [];

    (chat.documents || []).forEach(doc => {
        currentDocuments.push({
            name: doc.name,
            type: doc.type,
            size: doc.size,
            pages: doc.pages || 1,
            ocr: doc.ocr || false,
            url: null,
            documentId: doc.documentId || ""
        });
    });

    updateDocuments();

    getEl("messages").innerHTML = "";
    getEl("welcome").style.display = "none";

    (chat.messages || []).forEach(message => {
        if (message.role === "user") {
            addUserMessage(message.text);
        } else {
            addAIMessage(message.text, message.sources || []);
        }
    });

    renderHistory();

    requestAnimationFrame(() => {
        const chatScroll = getEl("chatScroll");
        chatScroll.scrollTop = chatScroll.scrollHeight;
    });
}

function deleteChat(event, chatId) {

    event.stopPropagation();

    const chat = chatHistory.find(item => item.id === chatId);
    if (!chat) return;

    const names = (chat.documents || []).map(doc => doc.name).join(", ");

    if (!confirm(`Delete this conversation?\n\n${names}`)) return;

    chatHistory = chatHistory.filter(item => item.id !== chatId);
    saveHistoryToStorage();

    if (currentChatId === chatId) {

        currentChatId = null;

        releaseDocumentUrls();
        currentDocuments = [];

        getEl("messages").innerHTML = "";
        getEl("welcome").style.display = "block";

        updateDocuments();
    }

    renderHistory();
}

function clearHistory() {

    if (!chatHistory.length) return;
    if (!confirm("Delete all recent chat history?")) return;

    chatHistory = [];
    localStorage.removeItem(STORAGE_KEY);

    renderHistory();
}

function newChat() {

    releaseDocumentUrls();

    currentDocuments = [];
    currentChatId = null;
    lastAnswer = "";
    lastQuestion = "";

    getEl("messages").innerHTML = "";
    getEl("question").value = "";
    getEl("suggestions").innerHTML = "";
    getEl("welcome").style.display = "block";

    updateDocuments();
    renderHistory();

    getEl("question").focus();
}

function clearCurrentChat() {

    const chat = findCurrentChat();
    if (!chat) return;

    chat.messages = [];
    chat.updatedAt = new Date().toISOString();

    getEl("messages").innerHTML = "";
    getEl("question").value = "";

    saveHistoryToStorage();
    renderHistory();
}


/* =========================
   SOURCE PREVIEW
========================= */

function showSourceByIndex(index) {
    const source = sourceRegistry[index];
    if (source) showSource(source);
}

function showSource(source) {

    const doc = currentDocuments.find(item => item.name === source.filename);

    getEl("sourceTitle").textContent = source.filename;
    getEl("sourcePage").textContent = `Page ${source.page || 1}`;
    getEl("sourceText").textContent = source.text || "No extracted text available.";

    const preview = getEl("preview");
    preview.innerHTML = "";

    if (!doc || !doc.url) {

        const note = document.createElement("div");
        note.style.cssText = "padding:40px;text-align:center;color:#7b849d;";
        note.textContent = "Upload this document again to preview the original page.";
        preview.appendChild(note);

    } else {

        const iframe = document.createElement("iframe");
        iframe.src = `${doc.url}#page=${source.page || 1}`;
        iframe.title = "Document page preview";
        preview.appendChild(iframe);
    }

    getEl("sourceModal").classList.add("open");
}

function closeSource(event) {

    const modal = getEl("sourceModal");

    if (event && event.target !== modal) return;

    modal.classList.remove("open");
}


/* =========================
   SUGGESTIONS
========================= */

function handleTyping() {

    const input = getEl("question");
    const box = getEl("suggestions");

    autoResize(input);

    const value = input.value.trim().toLowerCase();

    if (!value) {
        box.innerHTML = "";
        return;
    }

    const matches = suggestionPool
        .filter(item => item.toLowerCase().includes(value))
        .slice(0, 4);

    box.innerHTML = "";

    matches.forEach(item => {

        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion";
        button.textContent = item;

        button.addEventListener("mousedown", event => event.preventDefault());

        button.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();
            useSuggestion(item);
        });

        box.appendChild(button);
    });
}

function useSuggestion(text) {

    const input = getEl("question");

    input.value = text;
    autoResize(input);

    getEl("suggestions").innerHTML = "";

    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
}


/* =========================
   ENTER / SHIFT + ENTER
========================= */

function handleEnter(event) {

    if (event.key !== "Enter") return;

    // Do not send while an IME (Hindi/Kannada keyboard) is composing text
    if (event.isComposing || event.keyCode === 229) return;

    if (event.shiftKey) return;     // new line

    event.preventDefault();
    sendQuestion();
}

function autoResize(element) {
    element.style.height = "auto";
    element.style.height = Math.min(element.scrollHeight, 105) + "px";
}


/* =========================
   SHARE
========================= */

function shareChat() {

    const chat = findCurrentChat();
    if (!chat) return;

    const text = (chat.messages || [])
        .map(message => {
            const role = message.role === "user" ? "You" : "MedRAG";
            return `${role}:\n${message.text}`;
        })
        .join("\n\n");

    if (navigator.share) {
        navigator.share({ title: "MedRAG conversation", text: text }).catch(() => {});
    } else if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => alert("Conversation copied."));
    }
}


/* =========================
   THEME
========================= */

function changeTheme() {
    document.body.classList.toggle("dark");
}