/**
 * FRAUDGUARD - Main Client-Side JavaScript
 * File: static/js/script.js
 * Description: Form validation, responsive navigation, and UI interactivity.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Auto-dismiss alerts after 6 seconds
    const alerts = document.querySelectorAll('.alert');
    if (alerts.length > 0) {
        setTimeout(() => {
            alerts.forEach(alert => {
                alert.style.transition = 'opacity 0.5s ease';
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 500);
            });
        }, 6000);
    }

    // Input sanity checks on manual input
    const amountInput = document.getElementById('Amount');
    if (amountInput) {
        amountInput.addEventListener('input', (e) => {
            if (parseFloat(e.target.value) < 0) {
                e.target.value = 0;
            }
        });
    }

    const timeInput = document.getElementById('Time');
    if (timeInput) {
        timeInput.addEventListener('input', (e) => {
            if (parseFloat(e.target.value) < 0) {
                e.target.value = 0;
            }
        });
    }
});
