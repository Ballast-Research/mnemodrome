/**
 * Mnemodrome Admin UI - Shared JavaScript Utilities
 * Feature 013: MCP Config Web UI
 *
 * Provides:
 * - API client with authentication
 * - Toast notification system
 * - Form validation helpers
 * - Loading state management
 * - Utility functions
 */

// ============================================================================
// API Client
// ============================================================================

/**
 * Session expired or never logged in: go to the login form and come back.
 * (Feature 039 form login; API calls carry X-Requested-With so the server
 * answers 401 without a Basic-auth challenge.)
 */
function redirectIfLoggedOut(response) {
    if (response.status === 401) {
        const next = encodeURIComponent(window.location.pathname + window.location.search);
        window.location.href = `/admin/login?next=${next}`;
        throw new Error('Not logged in');
    }
}

/**
 * Make authenticated GET request to API endpoint
 * @param {string} endpoint - API endpoint path
 * @returns {Promise<any>} - Response data
 */
async function apiGet(endpoint) {
    try {
        const response = await fetch(endpoint, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'X-Requested-With': 'mnemodrome-admin',
            },
        });
        redirectIfLoggedOut(response);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API GET error:', error);
        throw error;
    }
}

/**
 * Make authenticated POST request to API endpoint
 * @param {string} endpoint - API endpoint path
 * @param {object} data - Request payload
 * @returns {Promise<any>} - Response data
 */
async function apiPost(endpoint, data = null) {
    try {
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-Requested-With': 'mnemodrome-admin',
            },
        };

        if (data !== null) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(endpoint, options);
        redirectIfLoggedOut(response);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API POST error:', error);
        throw error;
    }
}

// ============================================================================
// Toast Notification System
// ============================================================================

/**
 * Create toast container if it doesn't exist
 */
function ensureToastContainer() {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    return container;
}

/**
 * Show toast notification
 * @param {string} message - Notification message
 * @param {string} type - Toast type: 'success', 'error', 'warning', 'info'
 * @param {number} duration - Duration in milliseconds (0 = no auto-dismiss)
 */
function showToast(message, type = 'info', duration = 5000) {
    const container = ensureToastContainer();

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const titles = {
        success: 'Success',
        error: 'Error',
        warning: 'Warning',
        info: 'Information'
    };

    const header = document.createElement('div');
    header.className = 'toast-header';
    const title = document.createElement('span');
    title.className = 'toast-title';
    title.textContent = titles[type] || 'Notification';
    const closeBtn = document.createElement('button');
    closeBtn.className = 'toast-close';
    closeBtn.textContent = '\u00d7';
    closeBtn.addEventListener('click', () => toast.remove());
    header.appendChild(title);
    header.appendChild(closeBtn);
    const body = document.createElement('div');
    body.className = 'toast-message';
    body.textContent = message;
    toast.appendChild(header);
    toast.appendChild(body);

    container.appendChild(toast);

    // Auto-dismiss after duration
    if (duration > 0) {
        setTimeout(() => {
            toast.remove();
        }, duration);
    }
}

/**
 * Show success toast
 * @param {string} message - Success message
 */
function showSuccess(message) {
    showToast(message, 'success');
}

/**
 * Show error toast
 * @param {string} message - Error message
 */
function showError(message) {
    showToast(message, 'error', 8000); // Longer duration for errors
}

/**
 * Show warning toast
 * @param {string} message - Warning message
 */
function showWarning(message) {
    showToast(message, 'warning');
}

/**
 * Show info toast
 * @param {string} message - Info message
 */
function showInfo(message) {
    showToast(message, 'info');
}

// ============================================================================
// Loading State Management
// ============================================================================

let loadingOverlay = null;

/**
 * Show loading overlay
 * @param {string} message - Optional loading message
 */
function showLoading(message = 'Loading...') {
    if (!loadingOverlay) {
        loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'text-align: center; color: white;';
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        const msg = document.createElement('p');
        msg.style.cssText = 'margin-top: 1rem; font-weight: 500;';
        msg.textContent = message;
        wrapper.appendChild(spinner);
        wrapper.appendChild(msg);
        loadingOverlay.appendChild(wrapper);
        document.body.appendChild(loadingOverlay);
    }
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.remove();
        loadingOverlay = null;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format timestamp for display
 * @param {string} isoString - ISO 8601 timestamp
 * @returns {string} - Formatted timestamp
 */
function formatTimestamp(isoString) {
    if (!isoString) return 'N/A';

    try {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        // Relative time for recent events
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;

        // Absolute time for older events
        return date.toLocaleString();
    } catch (error) {
        console.error('Error formatting timestamp:', error);
        return isoString;
    }
}

/**
 * Format bytes as human-readable size
 * @param {number} bytes - Size in bytes
 * @returns {string} - Formatted size (e.g., "1.5 MB")
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    if (!bytes || isNaN(bytes)) return 'N/A';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format number with thousands separator
 * @param {number} num - Number to format
 * @returns {string} - Formatted number
 */
function formatNumber(num) {
    if (!num || isNaN(num)) return '0';
    return num.toLocaleString();
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Validate integer within range
 * @param {any} value - Value to validate
 * @param {number} min - Minimum value (inclusive)
 * @param {number} max - Maximum value (inclusive, optional)
 * @returns {object} - {valid: boolean, error: string}
 */
function validateInteger(value, min, max = null) {
    const num = parseInt(value, 10);

    if (isNaN(num)) {
        return { valid: false, error: 'Must be a valid integer' };
    }

    if (num < min) {
        return { valid: false, error: `Must be at least ${min}` };
    }

    if (max !== null && num > max) {
        return { valid: false, error: `Must be at most ${max}` };
    }

    return { valid: true, error: null };
}

/**
 * Validate boolean value
 * @param {any} value - Value to validate
 * @returns {object} - {valid: boolean, error: string}
 */
function validateBoolean(value) {
    if (typeof value === 'boolean') {
        return { valid: true, error: null };
    }

    if (typeof value === 'string') {
        const lower = value.toLowerCase();
        if (lower === 'true' || lower === 'false') {
            return { valid: true, error: null };
        }
    }

    return { valid: false, error: 'Must be true or false' };
}

/**
 * Validate string enum
 * @param {any} value - Value to validate
 * @param {string[]} options - Valid options
 * @returns {object} - {valid: boolean, error: string}
 */
function validateEnum(value, options) {
    if (options.includes(value)) {
        return { valid: true, error: null };
    }

    return {
        valid: false,
        error: `Must be one of: ${options.join(', ')}`
    };
}

/**
 * Validate non-empty string
 * @param {any} value - Value to validate
 * @returns {object} - {valid: boolean, error: string}
 */
function validateString(value) {
    if (typeof value === 'string' && value.trim().length > 0) {
        return { valid: true, error: null };
    }

    return { valid: false, error: 'Must be a non-empty string' };
}

/**
 * Validate configuration changes
 * @param {object} changes - Configuration changes
 * @param {object} schema - Configuration schema
 * @returns {object} - {valid: boolean, errors: string[]}
 */
function validateConfigChanges(changes, schema) {
    const errors = [];

    for (const [key, value] of Object.entries(changes)) {
        const varConfig = schema[key];

        if (!varConfig) {
            errors.push(`${key}: Unknown configuration variable`);
            continue;
        }

        if (!varConfig.editable) {
            errors.push(`${key}: Cannot edit (restart required)`);
            continue;
        }

        // Validation rules based on variable name patterns
        let validation;

        if (key.includes('ENABLED') || key.includes('AUTO_CAPTURE') || key.includes('FAIL_LOUD')) {
            validation = validateBoolean(value);
        } else if (key.includes('DAYS') || key.includes('SIZE_MB') || key.includes('HOURS') || key.includes('SLA_MS') || key.includes('PORT')) {
            const min = key.includes('HOURS') ? 0 : 1;
            const max = key.includes('PORT') ? 65535 : null;
            validation = validateInteger(value, min, max);
        } else if (key === 'LOG_LEVEL') {
            validation = validateEnum(value, ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']);
        } else {
            validation = validateString(value);
        }

        if (!validation.valid) {
            errors.push(`${key}: ${validation.error}`);
        }
    }

    return {
        valid: errors.length === 0,
        errors: errors
    };
}

// ============================================================================
// DOM Helpers
// ============================================================================

/**
 * Set active navigation link
 * @param {string} page - Current page identifier
 */
function setActiveNavLink(page) {
    const links = document.querySelectorAll('nav .nav-links a');
    links.forEach(link => {
        if (link.getAttribute('data-page') === page) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Create status indicator element
 * @param {string} status - Status: 'healthy', 'warning', 'error'
 * @returns {HTMLElement} - Status indicator element
 */
function createStatusIndicator(status) {
    const indicator = document.createElement('span');
    indicator.className = `status-indicator ${status}`;
    indicator.setAttribute('aria-label', status);
    return indicator;
}

/**
 * Create badge element
 * @param {string} text - Badge text
 * @param {string} type - Badge type: 'success', 'warning', 'error', 'info'
 * @returns {HTMLElement} - Badge element
 */
function createBadge(text, type = 'info') {
    const badge = document.createElement('span');
    badge.className = `badge badge-${type}`;
    badge.textContent = text;
    return badge;
}

// ============================================================================
// Auto-Refresh Helper
// ============================================================================

/**
 * Setup auto-refresh for a function
 * @param {Function} refreshFn - Function to call on each refresh
 * @param {number} intervalMs - Refresh interval in milliseconds
 * @returns {number} - Interval ID (use with clearInterval to stop)
 */
function setupAutoRefresh(refreshFn, intervalMs) {
    const intervalId = setInterval(async () => {
        try {
            await refreshFn();
        } catch (error) {
            console.error('Auto-refresh error:', error);
        }
    }, intervalMs);

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        clearInterval(intervalId);
    });

    return intervalId;
}

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Handle API error with user-friendly message
 * @param {Error} error - Error object
 * @param {string} context - Error context (e.g., "loading configuration")
 */
function handleApiError(error, context = 'performing action') {
    console.error(`Error ${context}:`, error);

    // Extract error message with fallback chain
    let message = error.detail || error.message || String(error);

    // Try to parse JSON error response
    try {
        const parsed = JSON.parse(message);
        if (parsed.detail) {
            if (typeof parsed.detail === 'object' && parsed.detail.errors) {
                message = parsed.detail.errors.join('\n');
            } else {
                message = String(parsed.detail);
            }
        } else if (parsed.message) {
            message = String(parsed.message);
        } else if (parsed.error) {
            message = String(parsed.error);
        }
    } catch (e) {
        // Not JSON, use as-is
    }

    // Ensure message is a string
    if (typeof message !== 'string') {
        message = String(message);
    }

    // Handle 401 Unauthorized
    if (message.includes('401') || message.includes('Unauthorized')) {
        showError('Authentication failed. Please refresh the page and log in again.');
        return;
    }

    // Handle 409 Conflict (version mismatch)
    if (message.includes('409') || message.includes('version mismatch')) {
        showError('Configuration was modified by another user. Please reload the page.');
        return;
    }

    showError(`Error ${context}: ${message}`);
}

// ============================================================================
// Dark Mode Management
// ============================================================================

/**
 * Toggle dark mode on/off
 */
function toggleDarkMode() {
    const body = document.body;
    const icon = document.getElementById('dark-mode-icon');

    body.classList.toggle('dark-mode');
    const isDarkMode = body.classList.contains('dark-mode');

    // Update icon
    if (icon) {
        icon.textContent = isDarkMode ? '☀️' : '🌙';
    }

    // Save preference to localStorage
    localStorage.setItem('darkMode', isDarkMode ? 'enabled' : 'disabled');
}

/**
 * Initialize dark mode from saved preference
 */
function initDarkMode() {
    const darkMode = localStorage.getItem('darkMode');
    const icon = document.getElementById('dark-mode-icon');

    if (darkMode === 'enabled') {
        document.body.classList.add('dark-mode');
        if (icon) icon.textContent = '☀️';
    } else {
        document.body.classList.remove('dark-mode');
        if (icon) icon.textContent = '🌙';
    }
}

// ============================================================================
// Initialization
// ============================================================================

// Initialize dark mode on page load
document.addEventListener('DOMContentLoaded', initDarkMode);

// Log when app.js is loaded
console.log('Mnemodrome Admin UI - app.js loaded');
