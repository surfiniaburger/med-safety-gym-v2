/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Basic safety check for generated HTML to prevent obvious malicious patterns.
 * While we use iframe sandboxing, this provides an extra layer of defense.
 */
export const validateHtmlSafety = (html: string): { safe: boolean; reason?: string } => {
    // Check for suspicious patterns that might try to break out of sandboxes
    // or perform sensitive actions.

    const suspiciousPatterns = [
        /\blocalStorage\b/i,
        /\bsessionStorage\b/i,
        /\bindindexedDB\b/i,
        /\bcookie\b/i,
        /\bfetch\s*\(/i,
        /\bXMLHttpRequest\b/i,
        /\bWebSocket\b/i,
        /\bwindow\.parent\b/i,
        /\bwindow\.top\b/i,
        /\beval\s*\(/i,
        /\bFunction\s*\(/i
    ];

    for (const pattern of suspiciousPatterns) {
        if (pattern.test(html)) {
            return {
                safe: false,
                reason: `Suspicious pattern detected: ${pattern.source}`
            };
        }
    }

    // Ensure it at least looks like HTML
    if (!html.toLowerCase().includes('<html') && !html.toLowerCase().includes('<!doctype')) {
        if (!html.toLowerCase().includes('<div') && !html.toLowerCase().includes('<style')) {
            return { safe: false, reason: 'Invalid HTML structure' };
        }
    }

    return { safe: true };
};
