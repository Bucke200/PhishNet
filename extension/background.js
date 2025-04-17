// PhishNet Detector - background.js

const BACKEND_URL = "https://phishnet-pavv.onrender.com"; // Your live backend API address
const PREDICT_ENDPOINT = `${BACKEND_URL}/predict`;
const REPORT_ENDPOINT = `${BACKEND_URL}/report`;

// Store notification data temporarily to link button clicks back to URLs
const notificationData = {};

// --- Helper Functions ---

async function checkUrl(tabId, url) {
    console.log(`Checking URL: ${url} for Tab ID: ${tabId}`);
    if (!url || !url.startsWith('http')) {
        console.log("Skipping check for non-http(s) URL.");
        return;
    }

    try {
        const response = await fetch(PREDICT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url }),
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log("Prediction result:", result);
        showNotification(url, result.prediction);

    } catch (error) {
        console.error("Error checking URL:", error);
        // Optionally show an error notification to the user
        // chrome.notifications.create({ ... });
    }
}

function showNotification(url, prediction) {
    const notificationId = `phishnet-${Date.now()}-${Math.random()}`;
    let options = {
        type: "basic",
        requireInteraction: true, // Keep notification until user interacts
        buttons: [
            { title: "Incorrect - It's Safe" }, // Button index 0
            { title: "Incorrect - It's Phishing" } // Button index 1
        ]
    };

    // --- CORRECTED LABEL MEANING (for urlset model): 0 = Legit, 1 = Phishing ---
    if (prediction === 1) { // Phishing detected (prediction is 1)
        options.title = "Phishing Warning!";
        options.message = `Potential phishing detected for:\n${url}`;
        options.iconUrl = "icons/icon-warning.png";
        options.priority = 2; // High priority
    } else { // Legit (prediction === 0)
        options.title = "URL Looks Safe";
        options.message = `This URL appears to be safe:\n${url}`;
        options.iconUrl = "icons/icon-safe.png";
        options.priority = 0; // Normal priority
    }

    // Store the URL associated with this notification ID for feedback reporting
    notificationData[notificationId] = { url: url, prediction: prediction };

    chrome.notifications.create(notificationId, options, (createdId) => {
        console.log(`Notification created with ID: ${createdId}`);
        // Set a timer to remove the notification data if not clicked (e.g., after 5 minutes)
        // to prevent memory leaks if notifications are dismissed without clicking buttons.
        chrome.alarms.create(`clearNotification_${createdId}`, { delayInMinutes: 5 });
    });
}

async function reportIncorrectPrediction(notificationId, buttonIndex) {
    const data = notificationData[notificationId];
    if (!data) {
        console.error(`No data found for notification ID: ${notificationId}`);
        return;
    }

    const originalUrl = data.url;
    // Determine the label the user is reporting based on the button clicked
    // Determine the label the user is reporting based on the button clicked
    // Button 0 ("Incorrect - It's Safe") means user reports label 0 (Legit)
    // Button 1 ("Incorrect - It's Phishing") means user reports label 1 (Phishing)
    // This needs to match the urlset convention (0=Legit, 1=Phishing)
    const reportedLabel = buttonIndex === 0 ? 0 : 1;

    console.log(`User reported prediction for ${originalUrl} as incorrect. Correct label reported by user: ${reportedLabel} (0=Legit, 1=Phishing)`);

    try {
        const response = await fetch(REPORT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: originalUrl, reported_label: reportedLabel }),
        });

        if (!response.ok) {
            throw new Error(`API report request failed: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log("Feedback report result:", result);
        // Optionally show a confirmation notification
        chrome.notifications.create({
             type: "basic",
             iconUrl: "icons/icon128.png", // Use main icon
             title: "Feedback Sent",
             message: "Thank you for your feedback!",
             priority: 0
        });


    } catch (error) {
        console.error("Error reporting feedback:", error);
        // Optionally show an error notification
    } finally {
        // Clean up the stored data for this notification
        delete notificationData[notificationId];
        // Clear the corresponding alarm
        chrome.alarms.clear(`clearNotification_${notificationId}`);
    }
}


// --- Event Listeners ---

// Listen for tab updates (navigation)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    // Check if the tab finished loading and has a URL
    if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith('http')) {
        checkUrl(tabId, tab.url);
    }
});

// Listen for notification button clicks
chrome.notifications.onButtonClicked.addListener((notificationId, buttonIndex) => {
    if (notificationId.startsWith('phishnet-')) {
        reportIncorrectPrediction(notificationId, buttonIndex);
    }
});

// Listen for notification close events (optional: clean up data if closed manually)
chrome.notifications.onClosed.addListener((notificationId, byUser) => {
     if (notificationId.startsWith('phishnet-') && notificationData[notificationId]) {
         console.log(`Notification ${notificationId} closed by user: ${byUser}. Cleaning up data.`);
         delete notificationData[notificationId];
         chrome.alarms.clear(`clearNotification_${notificationId}`);
     }
});

// Listener for alarms (used to clean up old notification data)
chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name.startsWith('clearNotification_')) {
        const notificationId = alarm.name.replace('clearNotification_', '');
        if (notificationData[notificationId]) {
            console.log(`Alarm triggered for ${notificationId}. Cleaning up stale notification data.`);
            delete notificationData[notificationId];
        }
    }
});


console.log("PhishNet Detector background script loaded.");
