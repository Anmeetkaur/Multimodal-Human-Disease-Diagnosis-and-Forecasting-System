/**
 * MediFuse API Client
 * Drop this <script> tag into your HTML — connects the frontend to FastAPI backend.
 * 
 * Usage: <script src="api.js"></script>
 * Then call: MediFuseAPI.predict(formData).then(result => ...)
 */

const MediFuseAPI = (() => {
  // ── CONFIG ──
  // Change this to your deployed backend URL, e.g. "https://your-server.com"
  const BASE_URL = "http://localhost:8000";
  let DEMO_MODE = false;  // Automatically set to true if backend is unreachable

  // Check if backend is alive on page load
  async function checkHealth() {
    try {
      const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        console.log("✅ MediFuse backend connected");
        DEMO_MODE = false;
        showConnectionStatus(true);
      }
    } catch {
      console.warn("⚠️ Backend unreachable — running in DEMO mode");
      DEMO_MODE = true;
      showConnectionStatus(false);
    }
  }

  function showConnectionStatus(connected) {
    // Remove existing banner
    document.getElementById("mf-status-banner")?.remove();
    if (!connected) {
      const banner = document.createElement("div");
      banner.id = "mf-status-banner";
      banner.style.cssText = `
        position: fixed; bottom: 16px; right: 16px; z-index: 9999;
        background: #fef3cd; border: 1px solid #e8a020; border-radius: 10px;
        padding: 10px 16px; font-size: 13px; color: #7a4a00;
        font-family: 'DM Sans', sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        display: flex; align-items: center; gap: 8px;
      `;
      banner.innerHTML = `⚠️ <strong>Demo Mode</strong> — Backend not connected. <a href="#" onclick="MediFuseAPI.retryConnection()" style="color:var(--blue);text-decoration:none">Retry</a>`;
      document.body.appendChild(banner);
    }
  }

  // ── DEMO DATA ──
  function getDemoResult(formData) {
    const name = formData.patient_name || "Demo Patient";
    return {
      success: true,
      report_id: "DEMO-" + Math.random().toString(36).slice(2, 8).toUpperCase(),
      patient_name: name,
      analysis_timestamp: new Date().toISOString(),
      processing_time_ms: 312,
      overall_risk_score_pct: 67.4,
      overall_risk_level: "Medium",
      conditions: [
        {
          disease: "Hypertensive Heart Disease", probability_pct: 74.2, risk_level: "High",
          specialty: "Cardiologist",
          xai_factors: [
            { feature: "BP Systolic 138 mmHg", source: "Blood", impact: "high", shap: +0.38 },
            { feature: "Age 42 years", source: "Tabular", impact: "high", shap: +0.24 },
            { feature: "Family history", source: "Tabular", impact: "medium", shap: +0.14 },
            { feature: "Symptom: chest pressure", source: "Text", impact: "medium", shap: +0.11 },
            { feature: "Active lifestyle", source: "Tabular", impact: "low", shap: -0.12 },
          ],
        },
        {
          disease: "Type 2 Diabetes", probability_pct: 62.1, risk_level: "Medium",
          specialty: "Endocrinologist",
          xai_factors: [
            { feature: "Fasting glucose 112 mg/dL", source: "Blood", impact: "high", shap: +0.32 },
            { feature: "BMI 27.4 (Overweight)", source: "Tabular", impact: "high", shap: +0.20 },
            { feature: "Sedentary lifestyle", source: "Tabular", impact: "medium", shap: +0.16 },
            { feature: "Symptom: fatigue", source: "Text", impact: "low", shap: +0.08 },
          ],
        },
        {
          disease: "Iron Deficiency Anemia", probability_pct: 54.8, risk_level: "Medium",
          specialty: "Hematologist",
          xai_factors: [
            { feature: "Hemoglobin 11.8 g/dL", source: "Blood", impact: "high", shap: +0.42 },
            { feature: "Female gender", source: "Tabular", impact: "medium", shap: +0.10 },
            { feature: "Symptom: fatigue", source: "Text", impact: "low", shap: +0.06 },
          ],
        },
        {
          disease: "Metabolic Syndrome", probability_pct: 33.5, risk_level: "Low",
          specialty: "General Physician",
          xai_factors: [
            { feature: "BMI 27.4", source: "Tabular", impact: "medium", shap: +0.15 },
            { feature: "Cholesterol 192 mg/dL", source: "Blood", impact: "low", shap: +0.08 },
          ],
        },
      ],
      modality_weights: { text: 0.26, tabular: 0.54, signal: 0.20 },
      encoder_info: {
        text: { model: "BERT", dim: 768, keywords: ["fatigue", "chest", "headache"] },
        tabular: { model: "MLP", dim: 256, bmi: 27.4 },
        signal: { model: "LSTM", dim: 128, anomaly_score: 0.18 },
        fusion: { method: "Cross-Attention", unified_dim: 512 },
      },
      recommendations: [
        { priority: "immediate", action: "ECG and echocardiogram within 7 days", specialist: "Cardiologist" },
        { priority: "soon", action: "HbA1c test within 2 weeks; dietitian consultation", specialist: "Endocrinologist" },
        { priority: "routine", action: "Iron supplementation; repeat CBC in 4 weeks", specialist: "Hematologist" },
      ],
      disclaimer: "This is an AI-assisted risk assessment only. Not a clinical diagnosis.",
      _demo: true,
    };
  }

  // ── API CALLS ──

  async function predict(formData) {
    if (DEMO_MODE) {
      await new Promise(r => setTimeout(r, 1200)); // Simulate latency
      return getDemoResult(formData);
    }
    const res = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });
    if (!res.ok) throw new Error(`Predict failed: ${res.statusText}`);
    return res.json();
  }

  async function encode(data) {
    if (DEMO_MODE) {
      return { success: true, text_encoder: { keywords_detected: ["fatigue", "pain"] }, _demo: true };
    }
    const res = await fetch(`${BASE_URL}/encode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  }

  async function getHospitals(city = "Delhi", specialty = "") {
    if (DEMO_MODE) {
      return {
        city,
        hospitals: [
          { name: "Apollo Hospital", address: "Greams Road, Chennai", distance_km: 1.2, rating: 4.7, specialties: ["Cardiology", "Endocrinology"], open: true },
          { name: "Govt. General Hospital", address: "Park Town, Chennai", distance_km: 2.8, rating: 4.2, specialties: ["General Medicine"], open: true },
          { name: "Frontier Heart Clinic", address: "Anna Nagar, Chennai", distance_km: 4.1, rating: 4.9, specialties: ["Cardiology"], open: true },
        ],
        _demo: true,
      };
    }
    const res = await fetch(`${BASE_URL}/hospitals?city=${encodeURIComponent(city)}&specialty=${encodeURIComponent(specialty)}`);
    return res.json();
  }

  async function login(email, password, role) {
    if (DEMO_MODE) {
      return { success: true, token: "demo-token", user: { email, role }, _demo: true };
    }
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, role }),
    });
    return res.json();
  }

  async function adminStats() {
    if (DEMO_MODE) return null;
    const res = await fetch(`${BASE_URL}/admin/stats`);
    return res.json();
  }

  async function retryConnection() {
    await checkHealth();
  }

  // ── HELPERS for rendering ──

  function renderRiskBadge(level) {
    const map = { High: "risk-high", "Medium-High": "risk-high", Medium: "risk-medium", Low: "risk-low" };
    const cls = map[level] || "risk-medium";
    return `<span class="risk-badge ${cls}"><span class="risk-dot"></span>${level}</span>`;
  }

  function renderConditionCards(conditions, container) {
    if (!container) return;
    container.innerHTML = "";
    conditions.forEach(c => {
      const pct = c.probability_pct;
      const barColor = pct >= 65
        ? "linear-gradient(90deg,#d63c3c,#f87575)"
        : "linear-gradient(90deg,#e8a020,#f5c95c)";
      const borderColor = pct >= 65 ? "var(--red)" : "var(--amber)";
      const card = document.createElement("div");
      card.className = "disease-card";
      card.style.borderLeft = `4px solid ${borderColor}`;
      card.innerHTML = `
        <div class="disease-name">
          ${c.disease}
          ${renderRiskBadge(c.risk_level)}
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width:${pct}%;background:${barColor}"></div>
        </div>
        <div class="prob-label">
          <span><strong>${pct}%</strong> probability</span>
          <span>🏥 ${c.specialty}</span>
        </div>
        <p style="font-size:12px;color:var(--text-3);margin-top:8px">
          Top factor: ${c.xai_factors?.[0]?.feature || "—"}
        </p>
        <button class="btn btn-secondary btn-sm" style="margin-top:10px"
          onclick="MediFuseAPI.showXAI(${JSON.stringify(c).replace(/"/g, '&quot;')})">
          Why? (SHAP)
        </button>
      `;
      container.appendChild(card);
    });
  }

  function showXAI(condition) {
    alert(`XAI for ${condition.disease}:\n\n` +
      condition.xai_factors.map(f => `${f.shap > 0 ? "▲" : "▼"} ${f.feature} (${f.source}, ${f.impact})`).join("\n"));
  }

  // Init
  checkHealth();

  return { predict, encode, getHospitals, login, adminStats, retryConnection, renderConditionCards, renderRiskBadge, showXAI };
})();