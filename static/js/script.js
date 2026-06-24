// Global variables
let currentFile = null;
let currentTheme = 'light';

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewSection = document.getElementById('previewSection');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const resultCard = document.getElementById('resultCard');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const themeText = document.getElementById('themeText');
const autoMode = document.getElementById('autoMode');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    setupEventListeners();
    checkSystemHealth();
});

// Theme Management
function initializeTheme() {
    // Check for saved theme preference or default to light
    const savedTheme = localStorage.getItem('theme') || 'light';
    const savedAutoMode = localStorage.getItem('autoMode') === 'true';
    
    autoMode.checked = savedAutoMode;
    
    if (savedAutoMode) {
        // Auto mode - detect system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            setTheme('dark');
        } else {
            setTheme('light');
        }
        
        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
            if (autoMode.checked) {
                setTheme(e.matches ? 'dark' : 'light');
            }
        });
    } else {
        setTheme(savedTheme);
    }
}

function setTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    document.body.className = theme === 'dark' ? 'dark-mode' : 'light-mode';
    
    // Update theme toggle button
    if (theme === 'dark') {
        themeIcon.className = 'fas fa-sun';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.className = 'fas fa-moon';
        themeText.textContent = 'Dark Mode';
    }
    
    // Save preference
    if (!autoMode.checked) {
        localStorage.setItem('theme', theme);
    }
}

function setupEventListeners() {
    // Theme toggle
    themeToggle.addEventListener('click', () => {
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    });
    
    // Auto mode toggle
    autoMode.addEventListener('change', () => {
        localStorage.setItem('autoMode', autoMode.checked);
        if (autoMode.checked) {
            // Switch to auto mode
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                setTheme('dark');
            } else {
                setTheme('light');
            }
        }
    });
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
}

// File handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, TIFF)');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    currentFile = file;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewSection.classList.remove('d-none');
        previewSection.classList.add('fade-in');
        
        // Hide any previous results or errors
        hideResults();
        hideError();
    };
    reader.readAsDataURL(file);
}

// API calls
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status !== 'healthy') {
            console.warn('System health check failed:', data);
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

async function analyzeImage() {
    if (!currentFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading
    showLoading();
    hideResults();
    hideError();
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            showError(result.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showError('Network error. Please try again.');
    } finally {
        hideLoading();
    }
}

// UI updates
function showLoading() {
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-in');
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingSection.style.display = 'none';
    analyzeBtn.disabled = false;
}

function displayResults(result) {
    const confidencePercent = parseFloat(result.confidence);
    
    // Update confidence meter
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceText.textContent = result.confidence;
    
    // Set confidence bar color based on tumor type
    const tumorClass = result.class_name.replace('_', '-');
    confidenceBar.className = `progress-bar tumor-${tumorClass}`;
    
    // Create result card content
    const resultHTML = `
        <div class="result-item tumor-${tumorClass}">
            <div class="result-icon">
                <i class="fas fa-brain"></i>
            </div>
            <div class="result-content">
                <h6>Tumor Type</h6>
                <p>${result.tumor_type}</p>
            </div>
        </div>
        
        <div class="result-item">
            <div class="result-icon" style="background-color: #007bff;">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="result-content">
                <h6>Symptoms</h6>
                <p>${result.symptoms}</p>
            </div>
        </div>
        
        <div class="result-item">
            <div class="result-icon" style="background-color: #28a745;">
                <i class="fas fa-pills"></i>
            </div>
            <div class="result-content">
                <h6>Treatment</h6>
                <p>${result.treatment}</p>
            </div>
        </div>
        
        <div class="result-item">
            <div class="result-icon" style="background-color: #fd7e14;">
                <i class="fas fa-chart-line"></i>
            </div>
            <div class="result-content">
                <h6>Cure Rate</h6>
                <p>${result.cure_rate}</p>
            </div>
        </div>
        
        <div class="result-item">
            <div class="result-icon" style="background-color: #6f42c1;">
                <i class="fas fa-exclamation-circle"></i>
            </div>
            <div class="result-content">
                <h6>Severity</h6>
                <p>${result.severity}</p>
            </div>
        </div>
        
        <div class="result-item">
            <div class="result-icon" style="background-color: #6c757d;">
                <i class="fas fa-info-circle"></i>
            </div>
            <div class="result-content">
                <h6>Description</h6>
                <p>${result.description}</p>
            </div>
        </div>
    `;
    
    resultCard.innerHTML = resultHTML;
    
    // Show results
    resultsSection.style.display = 'block';
    resultsSection.classList.add('slide-up');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.style.display = 'block';
    errorSection.classList.add('fade-in');
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorSection.style.display = 'none';
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + U to upload file
    if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
        event.preventDefault();
        fileInput.click();
    }
    
    // Ctrl/Cmd + Enter to analyze
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (currentFile) {
            analyzeImage();
        }
    }
    
    // Ctrl/Cmd + T to toggle theme
    if ((event.ctrlKey || event.metaKey) && event.key === 't') {
        event.preventDefault();
        themeToggle.click();
    }
});

// Add loading animation to buttons
function addLoadingToButton(button, text) {
    const originalText = button.innerHTML;
    button.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${text}`;
    button.disabled = true;
    return () => {
        button.innerHTML = originalText;
        button.disabled = false;
    };
}

// Add success animation
function addSuccessAnimation(element) {
    element.classList.add('animate__animated', 'animate__pulse');
    setTimeout(() => {
        element.classList.remove('animate__animated', 'animate__pulse');
    }, 1000);
}

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        setTheme,
        processFile,
        analyzeImage,
        displayResults,
        showError
    };
} 