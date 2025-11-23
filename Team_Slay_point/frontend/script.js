const ADMIN_PASSWORD = "admin123"; // temp

// Role selection
function selectRole(role) {
    document.getElementById("role_selection").classList.add("hidden");
    if (role === "admin") {
        document.getElementById("admin_login").classList.remove("hidden");
    } else {
        document.getElementById("user_dashboard").classList.remove("hidden");
    }
}

// Admin login
function loginAdmin() {
    const password = document.getElementById("admin_password").value;
    const result = document.getElementById("admin_login_result");
    if (password === ADMIN_PASSWORD) {
        document.getElementById("admin_login").classList.add("hidden");
        document.getElementById("admin_dashboard").classList.remove("hidden");
        result.innerText = "";
    } else {
        result.innerText = "Incorrect password!";
    }
}

// Press Enter to login
function checkEnter(event, funcName) {
    if (event.key === "Enter") {
        window[funcName]();
    }
}

// Dummy predict function (interactive)
function predictTransaction(role) {
    const transaction = role === "admin" ?
        document.getElementById("transaction_input_admin").value :
        document.getElementById("transaction_input_user").value;

    if (!transaction) return;

    const categories = ["Food & Dining", "Shopping", "Fuel", "Groceries", "Bills & Utilities", "Travel", "Entertainment", "Rent"];
    
    // Simple keyword match for interactivity
    let category = "Unknown";
    const lower = transaction.toLowerCase();
    if (lower.includes("starbucks") || lower.includes("pizza") || lower.includes("kfc")) category = "Food & Dining";
    else if (lower.includes("amazon") || lower.includes("flipkart")) category = "Shopping";
    else if (lower.includes("petrol") || lower.includes("fuel") || lower.includes("shell")) category = "Fuel";

    const confidence = (Math.random() * 0.3 + 0.7).toFixed(2);

    const resultId = role === "admin" ? "prediction_result_admin" : "prediction_result_user";
    document.getElementById(resultId).innerText = `Category: ${category}, Confidence: ${confidence}`;
}

// Dummy Add Transaction
function addTransaction() {
    const trans = document.getElementById("new_transaction").value;
    const cat = document.getElementById("new_category").value;
    if (!trans || !cat) {
        document.getElementById("add_result").innerText = "Please enter transaction and category!";
        return;
    }
    document.getElementById("add_result").innerText = `Added: "${trans}" → ${cat}`;
}

// Dummy Update Category
function updateCategory() {
    const oldCat = document.getElementById("old_category").value;
    const newCat = document.getElementById("new_category_name").value;
    if (!oldCat || !newCat) {
        document.getElementById("update_result").innerText = "Enter both old and new category!";
        return;
    }
    document.getElementById("update_result").innerText = `Category "${oldCat}" updated to "${newCat}"`;
}
