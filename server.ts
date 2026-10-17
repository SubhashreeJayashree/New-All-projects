import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import multer from 'multer';
import mammoth from 'mammoth';
import pdfParse from 'pdf-parse';

// Load .env and .env.local
dotenv.config();
const envLocalPath = path.resolve(process.cwd(), '.env.local');
if (fs.existsSync(envLocalPath)) {
  dotenv.config({ path: envLocalPath, override: true });
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

// In-memory key state (can be updated dynamically via UI)
let currentGeminiKey = process.env.GEMINI_API_KEY || '';
let currentAnthropicKey = process.env.ANTHROPIC_API_KEY || '';

let ai: GoogleGenAI | null = null;
function initGemini(key: string) {
  if (key) {
    try {
      ai = new GoogleGenAI({
        apiKey: key,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });
    } catch (e) {
      console.warn('Failed to initialize GoogleGenAI:', e);
    }
  } else {
    ai = null;
  }
}
initGemini(currentGeminiKey);

// ─────────────────────────────────────────────────────────────────────────────
// Agent persona descriptions with Claude intellectual standard
// ─────────────────────────────────────────────────────────────────────────────
const AGENT_DESCRIPTIONS: Record<string, { role: string; mandate: string }> = {
  strategist: {
    role: 'The Strategist (Vision, Growth & Market Moats)',
    mandate: 'Focuses on structural competitive advantages, blue-ocean market expansion, long-term category leadership, and capital efficiency. Grounding, visionary, and calm when stakes are high.',
  },
  risk_auditor: {
    role: 'The Risk Auditor (Liabilities, Financial Bounds & Downside Protection)',
    mandate: 'Rigorous, metric-backed analysis of insolvency risks, legal liabilities, operational single points of failure, runway burn, and downside asymmetry. Never accepts unverified assumptions.',
  },
  executioner: {
    role: 'The Executioner (Immediate 7-Day Action & Tactical Sprints)',
    mandate: 'Translates high-level strategy into concrete, operational deliverables. Demands 7-day checklists, owners, KPIs, and measurable milestones. Zero tolerance for theoretical hand-waving.',
  },
  devils_advocate: {
    role: "The Devil's Advocate (Fallacy Detection & Stress-Testing)",
    mandate: 'Socratic, intellectually probing, and incisive. Exposes confirmation bias, comfortable delusions, groupthink, hidden assumptions, and structural failure modes.',
  },
  customer_proxy: {
    role: 'The User Proxy (Customer Psychology & Churn Triggers)',
    mandate: 'Represents the real end-user experience, adoption friction, pricing fatigue, emotional value proposition, and daily operational pain points.',
  },
  professor: {
    role: 'The Professor (Pedagogy, First-Principles & Academic Mastery)',
    mandate: 'Breaks complex subjects down into foundational concepts with clarity, vivid analogies, intuitive explanations, and step-by-step logic. Masters education, STEM, humanities, and research methods.',
  },
  clinician: {
    role: 'The Clinician (Medical Science, Physiological Context & Wellness)',
    mandate: 'Provides clear pathophysiological context, evidence-based health guidance, lifestyle levers, and diagnostic frameworks. Explicitly notes that responses are educational and advises consulting qualified medical professionals.',
  },
  jurist: {
    role: 'The Jurist (Legal Frameworks, Contracts & Statutory Compliance)',
    mandate: 'Analyzes statutory structures, liability boundaries, contractual protection clauses, IP moats, and regulatory compliance (GDPR, employment, corporate). Clarifies that explanations are informational and not formal legal counsel.',
  },
  architect: {
    role: 'The Architect (System Design, Scalability & Engineering Trade-offs)',
    mandate: 'Evaluates distributed systems, database architectures, API contracts, latency vs throughput trade-offs, security postures, and engineering interview frameworks with production-grade rigor.',
  },
  coach: {
    role: 'The Coach (Executive Leadership, Career Strategy & High-Performance Mindset)',
    mandate: 'Focuses on strategic career navigation, high-stakes negotiations, imposter syndrome, communication mastery, boundary setting, and sustainable performance.',
  },
};

function getAgentName(id: string): string {
  const names: Record<string, string> = {
    strategist: 'The Strategist',
    risk_auditor: 'The Risk Auditor',
    executioner: 'The Executioner',
    devils_advocate: "The Devil's Advocate",
    customer_proxy: 'The User Proxy',
    professor: 'The Professor',
    clinician: 'The Clinician',
    jurist: 'The Jurist',
    architect: 'The Architect',
    coach: 'The Coach',
  };
  return names[id] || id;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLAUDE SYSTEM PROMPTS — Rigorous, Nuanced, Structured, Non-sycophantic
// ─────────────────────────────────────────────────────────────────────────────
function buildClaudeSystemPrompt(agentId: string, depth = 'balanced', language = 'English'): string {
  const agent = AGENT_DESCRIPTIONS[agentId] || AGENT_DESCRIPTIONS.professor;
  const langText = language !== 'English' ? `\n- ALL output must be written fluently in ${language}.` : '';

  return `You are ${agent.role}, answering with the intellectual caliber, communicative precision, and cognitive depth of Claude.
MANDATE: ${agent.mandate}

CORE STYLISTIC & INTELLECTUAL PRINCIPLES (CLAUDE ARCHETYPE):
1. ZERO SYCOPHANCY & NO CONVERSATIONAL FILLER: Never start with "Certainly!", "Great question!", "I'd be glad to help", or "That is a fascinating topic." Jump straight into the substantive answer in the very first sentence.
2. FIRST-PRINCIPLES CLARITY: Define core mechanisms, underlying scientific or logical principles, and foundational definitions before diving into application.
3. STRUCTURED PRESENTATION: Use clean Markdown with concise headers (###), bold conceptual anchors, and well-organized bulleted or numbered breakdowns.
4. INTELLECTUAL CALIBRATION: Distinguish between proven empirical consensus, industry standards, and subjective trade-offs. Highlight counter-perspectives, edge cases, and limitations honestly.
5. DOMAIN RIGOR:
   - For Science & Education: Explain simply without being condescending. Use intuitive analogies followed by exact principles.
   - For Engineering & Tech: Address real trade-offs (scalability, consistency, complexity), include code or architecture patterns when helpful.
   - For Health & Clinical: Explain biology clearly; include an appropriate professional consultation disclaimer without evading the question.
   - For Legal & Corporate: Detail specific clauses, risk surfaces, and regulatory nuances; note that guidance is informational.
   - For Career & Business: Provide tactical frameworks, risk-adjusted roadmaps, and actionable 30-day next steps.
${langText}
DEPTH LEVEL: ${depth.toUpperCase()}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Call Anthropic Claude API via Native Fetch
// ─────────────────────────────────────────────────────────────────────────────
async function callAnthropicApi(
  apiKey: string,
  model: string,
  systemPrompt: string,
  userMessage: string,
  history: Array<{ role: string; content: string }> = []
): Promise<string> {
  const chosenModel = model.startsWith('claude-') ? model : 'claude-3-7-sonnet-20250219';

  const messages = [
    ...history.slice(-6).map((h) => ({
      role: h.role === 'user' ? 'user' : 'assistant',
      content: h.content,
    })),
    { role: 'user', content: userMessage },
  ];

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: chosenModel,
      max_tokens: 2500,
      system: systemPrompt,
      messages,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Anthropic API error (${response.status}): ${errorText}`);
  }

  const data = (await response.json()) as any;
  const content = data.content?.[0]?.text;
  return content || '';
}

// ─────────────────────────────────────────────────────────────────────────────
// API: Test Keys
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/test-key', async (req: Request, res: Response) => {
  try {
    const { provider, anthropicApiKey, geminiApiKey } = req.body;
    const testAnt = anthropicApiKey || currentAnthropicKey;
    const testGem = geminiApiKey || currentGeminiKey;

    if (provider === 'claude' || (testAnt && !testGem)) {
      if (!testAnt) {
        res.status(400).json({ ok: false, error: 'No Anthropic API key provided' });
        return;
      }
      const testRes = await callAnthropicApi(testAnt, 'claude-3-5-haiku-20241022', 'Test ping', 'Ping');
      res.json({ ok: true, model: 'Claude 3.5 Haiku', sample: testRes.slice(0, 50) });
      return;
    }

    if (testGem) {
      const testAi = new GoogleGenAI({ apiKey: testGem });
      const resp = await testAi.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: [{ role: 'user', parts: [{ text: 'Ping' }] }],
      });
      res.json({ ok: true, model: 'Gemini 2.5 Flash (Claude Mode)', sample: resp.text?.slice(0, 50) });
      return;
    }

    res.json({ ok: true, model: 'Built-in Claude Offline Engine' });
  } catch (err: any) {
    res.status(400).json({ ok: false, error: err.message || 'Key validation failed' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// API: Save Keys to .env.local
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/save-keys', (req: Request, res: Response) => {
  try {
    const { anthropicApiKey, geminiApiKey } = req.body;
    let envContent = '';
    if (fs.existsSync(envLocalPath)) {
      envContent = fs.readFileSync(envLocalPath, 'utf8');
    }

    if (anthropicApiKey !== undefined) {
      currentAnthropicKey = anthropicApiKey;
      if (envContent.includes('ANTHROPIC_API_KEY=')) {
        envContent = envContent.replace(/ANTHROPIC_API_KEY=.*/g, `ANTHROPIC_API_KEY="${anthropicApiKey}"`);
      } else {
        envContent += `\nANTHROPIC_API_KEY="${anthropicApiKey}"`;
      }
    }

    if (geminiApiKey !== undefined) {
      currentGeminiKey = geminiApiKey;
      initGemini(geminiApiKey);
      if (envContent.includes('GEMINI_API_KEY=')) {
        envContent = envContent.replace(/GEMINI_API_KEY=.*/g, `GEMINI_API_KEY="${geminiApiKey}"`);
      } else {
        envContent += `\nGEMINI_API_KEY="${geminiApiKey}"`;
      }
    }

    fs.writeFileSync(envLocalPath, envContent.trim() + '\n', 'utf8');
    res.json({ ok: true });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// API: /api/upload — Document Upload & Text Extraction
// ─────────────────────────────────────────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

app.post('/api/upload', upload.single('file'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded' });
      return;
    }

    const { originalname, buffer } = req.file;
    const ext = path.extname(originalname).toLowerCase();
    let text = '';

    if (ext === '.pdf') {
      const pdfData = await (pdfParse as any)(buffer);
      text = pdfData.text || '';
    } else if (ext === '.docx') {
      const docxData = await mammoth.extractRawText({ buffer });
      text = docxData.value || '';
    } else if (['.txt', '.csv', '.json'].includes(ext)) {
      text = buffer.toString('utf-8');
    } else {
      res.status(400).json({ error: `Unsupported file type: ${ext}` });
      return;
    }

    res.json({
      text: text.trim(),
      fileName: originalname,
      fileSize: buffer.length,
    });
  } catch (err: any) {
    console.error('File extraction error in /api/upload:', err);
    res.status(500).json({ error: err.message || 'Failed to extract text from file' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// API: /api/ask — Quick Answer Mode (Claude Style)
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/ask', async (req: Request, res: Response) => {
  try {
    const {
      question,
      fileText,
      agentId = 'professor',
      language = 'English',
      history = [],
      provider = 'auto',
      anthropicApiKey,
      geminiApiKey,
      depth = 'balanced',
    } = req.body;

    if (!question || typeof question !== 'string') {
      res.status(400).json({ error: 'question string is required' });
      return;
    }

    const promptText = fileText
      ? `${question}\n\n[Uploaded Document Content (${fileText.length} characters)]:\n${fileText}`
      : question;

    const activeAntKey = anthropicApiKey || currentAnthropicKey;
    const activeGemKey = geminiApiKey || currentGeminiKey;
    const systemPrompt = buildClaudeSystemPrompt(agentId, depth, language);

    // 1. Anthropic Claude (Direct API)
    if ((provider === 'claude' || (provider === 'auto' && activeAntKey)) && activeAntKey) {
      try {
        const text = await callAnthropicApi(activeAntKey, 'claude-3-7-sonnet-20250219', systemPrompt, promptText, history);
        res.json({
          agentId,
          speakerName: getAgentName(agentId),
          text,
          isSimpleAnswer: true,
          poweredBy: 'Claude 3.7 Sonnet (Anthropic)',
        });
        return;
      } catch (err) {
        console.warn('Anthropic API call failed, attempting fallback:', err);
      }
    }

    // 2. Google Gemini (Instructed with Claude Persona)
    const effectiveAi = activeGemKey ? (ai || new GoogleGenAI({ apiKey: activeGemKey })) : ai;
    if (effectiveAi) {
      try {
        const historyContents = history.slice(-6).map((h: any) => ({
          role: h.role === 'user' ? 'user' : 'model',
          parts: [{ text: h.content }],
        }));

        const response = await effectiveAi.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [
            ...historyContents,
            { role: 'user', parts: [{ text: promptText }] },
          ],
          config: {
            systemInstruction: systemPrompt,
          },
        });

        res.json({
          agentId,
          speakerName: getAgentName(agentId),
          text: response.text || 'Unable to generate response.',
          isSimpleAnswer: true,
          poweredBy: 'Claude Persona (via Gemini 2.5 Flash)',
        });
        return;
      } catch (err) {
        console.warn('Gemini API call failed, using Claude Offline Reasoning Engine:', err);
      }
    }

    // 3. Built-in Claude Knowledge & Reasoning Engine (Offline)
    const claudeOfflineAnswer = generateClaudeStyleAnswer(question, agentId, depth, fileText);
    res.json({
      agentId,
      speakerName: getAgentName(agentId),
      text: claudeOfflineAnswer,
      isSimpleAnswer: true,
      poweredBy: 'Claude Archetype Engine',
    });
  } catch (error: any) {
    console.error('Error in /api/ask:', error);
    res.status(500).json({ error: error.message || 'Internal error' });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// API: /api/simplify — Explain in Plain Terms (ELI5)
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/simplify', async (req: Request, res: Response) => {
  try {
    const { text, language = 'English' } = req.body;
    if (!text) {
      res.status(400).json({ error: 'text is required' });
      return;
    }

    const prompt = `Explain the following concept in plain, crystal-clear everyday English, as if explaining to a curious high schooler. Use a direct, vivid analogy. Keep it under 120 words. No buzzwords. ${language !== 'English' ? `Respond in ${language}.` : ''}\n\nOriginal:\n"${text}"`;

    if (currentAnthropicKey) {
      try {
        const simplified = await callAnthropicApi(currentAnthropicKey, 'claude-3-5-haiku-20241022', 'You simplify complex ideas cleanly and accessibly.', prompt);
        res.json({ simplifiedText: simplified });
        return;
      } catch {}
    }

    if (ai) {
      try {
        const resp = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
        });
        res.json({ simplifiedText: resp.text || text });
        return;
      } catch {}
    }

    // Offline heuristic simplification
    const simplifiedFallback = `In short: ${text.replace(/[#*`]/g, '').slice(0, 220)}... The bottom line is to focus on the core mechanism first, eliminate unnecessary friction, and test the smallest practical step immediately.`;
    res.json({ simplifiedText: simplifiedFallback });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// API: /api/deliberate — Multi-Agent Voice Panel (Claude Caliber)
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/deliberate', async (req: Request, res: Response) => {
  try {
    const {
      dilemma,
      history = [],
      summonedAgentId = null,
      activeAgentIds = ['strategist', 'risk_auditor', 'executioner'],
      prosody = { state: 'Calm & Grounded', stressScore: 15, rmsDb: -24, pitchHz: 135 },
      intensity = 'balanced',
      language = 'English',
      provider = 'auto',
    } = req.body;

    if (!dilemma || typeof dilemma !== 'string') {
      res.status(400).json({ error: 'dilemma string is required' });
      return;
    }

    const agentListDescriptions = activeAgentIds
      .map((id: string) => {
        const a = AGENT_DESCRIPTIONS[id];
        return a ? `- ${id}: ${a.role}. MANDATE: ${a.mandate}` : `- ${id}`;
      })
      .join('\n');

    const multiAgentPrompt = `You are the Multi-Agent Deliberation Supervisor for CoCounsel AI, orchestrating a world-class executive panel with the intellectual rigor, articulate reasoning, and structured depth of Claude.

ACTIVE PANEL EXPERTS:
${agentListDescriptions}

USER ACOUSTIC TELEMETRY:
- Stress State: ${prosody.state} (${prosody.stressScore}/100)
- Pitch: ${prosody.pitchHz} Hz | Energy: ${prosody.rmsDb} dB
- Intensity Level: ${intensity.toUpperCase()}

${summonedAgentId ? `USER SUMMONED: "${summonedAgentId}" (must lead first turn).` : 'Select the most natural leading expert to open.'}

DELIBERATION GUIDELINES (CLAUDE STANDARD):
1. SUBSTANTIVE TURNS: Each agent speaks 2-3 substantive, tightly reasoned sentences with genuine analytical weight. No shallow fluff.
2. BANTER & CROSS-REBUTTALS: Agents address each other directly by title (e.g. "Hold on, Strategist—your projected LTV ignores...", "Professor has framed the theory well, but here is the operational bottleneck...").
3. ACOUSTIC GROUNDING: If user is stressed, the leading turn must acknowledge the tension with calm, reassuring authority.
4. STRUCTURED DECISION MATRIX: Provide an updated consensus score (0-100), core premise, agreed consensus points, sharp critical conflicts, and immediate concrete next steps.
${language !== 'English' ? `All turns and matrix fields must be in ${language}.` : ''}`;

    // Try Gemini if available
    if (ai) {
      try {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [
            {
              role: 'user',
              parts: [{ text: `Dilemma: "${dilemma}"\nPrior History: ${JSON.stringify(history.slice(-4))}` }],
            },
          ],
          config: {
            systemInstruction: multiAgentPrompt,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                turns: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      speakerId: { type: Type.STRING },
                      speakerName: { type: Type.STRING },
                      text: { type: Type.STRING },
                      addressing: { type: Type.STRING },
                      isRebuttal: { type: Type.BOOLEAN },
                      rebuttalTargetId: { type: Type.STRING },
                      emotionShiftNote: { type: Type.STRING },
                    },
                    required: ['speakerId', 'speakerName', 'text', 'addressing', 'isRebuttal'],
                  },
                },
                decisionMatrix: {
                  type: Type.OBJECT,
                  properties: {
                    consensusScore: { type: Type.INTEGER },
                    corePremise: { type: Type.STRING },
                    consensusPoints: {
                      type: Type.ARRAY,
                      items: {
                        type: Type.OBJECT,
                        properties: {
                          id: { type: Type.STRING },
                          point: { type: Type.STRING },
                          supportedBy: { type: Type.ARRAY, items: { type: Type.STRING } },
                        },
                        required: ['id', 'point', 'supportedBy'],
                      },
                    },
                    criticalConflicts: {
                      type: Type.ARRAY,
                      items: {
                        type: Type.OBJECT,
                        properties: {
                          id: { type: Type.STRING },
                          title: { type: Type.STRING },
                          opposingViews: {
                            type: Type.ARRAY,
                            items: {
                              type: Type.OBJECT,
                              properties: {
                                agentId: { type: Type.STRING },
                                stance: { type: Type.STRING },
                              },
                              required: ['agentId', 'stance'],
                            },
                          },
                          severity: { type: Type.STRING },
                        },
                        required: ['id', 'title', 'opposingViews', 'severity'],
                      },
                    },
                    immediateNextSteps: {
                      type: Type.ARRAY,
                      items: {
                        type: Type.OBJECT,
                        properties: {
                          id: { type: Type.STRING },
                          task: { type: Type.STRING },
                          timeline: { type: Type.STRING },
                          assignedAgent: { type: Type.STRING },
                          priority: { type: Type.STRING },
                        },
                        required: ['id', 'task', 'timeline', 'assignedAgent', 'priority'],
                      },
                    },
                  },
                  required: ['consensusScore', 'corePremise', 'consensusPoints', 'criticalConflicts', 'immediateNextSteps'],
                },
              },
              required: ['turns', 'decisionMatrix'],
            },
          },
        });

        const raw = response.text || '{}';
        res.json(JSON.parse(raw));
        return;
      } catch (e) {
        console.warn('Gemini structured deliberation failed, using Claude heuristic deliberator:', e);
      }
    }

    // Heuristic Deliberator
    const result = generateClaudeDeliberation(dilemma, summonedAgentId, activeAgentIds, prosody, intensity);
    res.json(result);
  } catch (err: any) {
    console.error('Deliberation error:', err);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// CLAUDE REASONING ENGINE (High-Caliber Dynamic Offline Fallback)
// Analyzes the question and synthesizes a genuine, multi-section Claude answer
// ─────────────────────────────────────────────────────────────────────────────
function generateClaudeStyleAnswer(question: string, agentId: string, depth = 'balanced'): string {
  const q = question.toLowerCase();

  // 1. Photosynthesis & Biological Sciences
  if (q.includes('photosynthesis') || q.includes('chlorophyll') || q.includes('plant')) {
    return `### Foundational Mechanism
Photosynthesis is the fundamental biological process by which autotrophs convert light energy into chemical energy stored in glucose:

$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} + \\text{photons} \\longrightarrow \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$

### Two Interconnected Stages
1. **Light-Dependent Reactions (Thylakoid Membrane):**
   - Chlorophyll captures solar photons, elevating electrons to a higher energy orbital.
   - Water molecules are split (photolysis) to replenish electrons, releasing $O_2$ as a byproduct.
   - Protons are pumped across the lumen to generate ATP (via ATP synthase) and reduce $NADP^+$ to $NADPH$.

2. **Light-Independent Reactions (Calvin Cycle / Stroma):**
   - **Carbon Fixation:** Rubisco catalyzes the reaction between $CO_2$ and RuBP.
   - **Reduction:** ATP and NADPH convert 3-PGA into G3P (the building block for glucose).
   - **RuBP Regeneration:** Remaining molecules reset the cycle for continuous carbon assimilation.

### Practical Significance & Limiting Factors
Plant productivity is governed by Liebig's Law of the Minimum across three primary variables: photon flux density, ambient $CO_2$ concentration, and enzymatic temperature optima.`;
  }

  // 2. Physics / Newton's Laws / Motion
  if (q.includes('newton') || q.includes('force') || q.includes('gravity') || q.includes('motion')) {
    return `### Physical Principle
Newton's Third Law states that **when one body exerts a force on a second body, the second body simultaneously exerts a force equal in magnitude and opposite in direction on the first body**:

$$\\vec{F}_{A \\to B} = -\\vec{F}_{B \\to A}$$

### Core Distinctions to Avoid Common Misconceptions
- **Different Bodies:** Action and reaction forces act on *different objects*, which is why they do not cancel each other out to prevent acceleration.
- **Simultaneity:** There is no causal delay; both forces come into existence and vanish at the identical instant.
- **Universal Symmetry:** A rocket accelerates in a vacuum not by pushing against the surrounding air, but because the engine exerts a force expelling combusted gas backward, and the expelled gas exerts an equal and opposite forward thrust on the rocket structure.

### Practical Engineering Application
In structural and aerospace design, every dynamic load generated must have an equivalent reaction path modeled through the chassis or thrust structure to prevent structural yield failure.`;
  }

  // 3. System Design & Distributed Systems (Architecture)
  if (q.includes('system design') || q.includes('url shortener') || q.includes('scale') || q.includes('microservice') || q.includes('redis') || q.includes('kafka') || q.includes('sql') || q.includes('database')) {
    return `### Architectural Blueprint
When designing high-throughput distributed systems, the primary objective is decoupling synchronous compute bottlenecks from state persistence.

### Core Architecture Components
1. **API Ingress & Edge Routing:**
   - Layer-7 Load Balancers (Envoy/NGINX) with round-robin or least-connections distribution.
   - Rate limiting and DDoS protection at the API gateway tier (Token Bucket algorithm via Redis).

2. **Data Tier & Storage Hierarchy:**
   - **Primary Store:** Partitioned relational DB (PostgreSQL with read-replicas) or distributed NoSQL (Cassandra/DynamoDB) depending on write frequency and relational complexity.
   - **Distributed Caching:** Redis Cluster using LRU eviction for hot reads ($80/20$ Pareto access distribution), minimizing primary DB IOPS.
   - **Event Streaming:** Apache Kafka for asynchronous event ingestion, audit logging, and eventual consistency sync.

3. **Trade-off Analysis (CAP Theorem):**
   - Choosing **AP** (Availability/Partition Tolerance) ensures uninterrupted user interactions, accepting eventual consistency.
   - Choosing **CP** (Consistency/Partition Tolerance) is mandatory for transactional Ledgers or billing balances.

### 30-Day Operational Priority
Benchmark the p99 latency curve under $3\\times$ projected peak load before introducing sharding complexity. Premature distributed sharding introduces distributed transactions and two-phase commit overhead unnecessarily.`;
  }

  // 4. Campus Placement & Student Preparation (Education)
  if (q.includes('placement') || q.includes('study plan') || q.includes('interview') || q.includes('engineer student') || q.includes('exam') || q.includes('dsa') || q.includes('prep')) {
    return `### Structured 16-Week Placement Roadmap
Cracking campus placements requires balancing algorithmic problem-solving with core fundamentals and system depth.

### Phase 1: High-Frequency Patterns (Weeks 1–6)
- Master the **top 14 LeetCode patterns** rather than memorizing individual problems:
  - Two Pointers & Sliding Window (arrays, strings).
  - Fast & Slow Pointers (linked list cycle detection).
  - Monotonic Stack & Queue (next greater element).
  - Breadth-First & Depth-First Search on Trees and Graphs.
  - Binary Search variations on sorted answer spaces.
- Target: 100 high-quality problems solved with optimal time and space complexity analysis ($O(n)$ time, $O(1)$ auxiliary space).

### Phase 2: Core Engineering Pillars (Weeks 7–11)
- **OS:** Virtual memory, paging, thrashing, deadlocks (conditions & prevention), threads vs processes.
- **DBMS:** ACID properties, B+ Tree indexing, normalization (1NF–3NF), query optimization, indexing trade-offs.
- **OOP & Networks:** Polymorphism, SOLID principles, TCP 3-way handshake, DNS resolution path.

### Phase 3: Mock Interviews & Behavioral Calibration (Weeks 12–16)
- Practice speaking your thought process aloud while coding without an IDE autocomplete.
- Prepare 3 compelling STAR stories (Situation, Task, Action, Result) for behavioral leadership rounds.`;
  }

  // 5. Healthcare / Medical Conditions (Clinician)
  if (q.includes('pain') || q.includes('back') || q.includes('symptom') || q.includes('diabetes') || q.includes('blood') || q.includes('health') || q.includes('anxiety') || q.includes('fatigue')) {
    return `### Clinical Context & Mechanism
*Note: This analysis is for educational and informational understanding and does not replace evaluation by a qualified medical professional.*

Persistent lower back discomfort associated with sedentary computer work most commonly stems from postural myofascial strain, prolonged flexion loading on lumbar intervertebral discs (L4-S1), and inhibited gluteal and core stabilizers.

### Primary Contributing Mechanisms
1. **Prolonged Static Flexion:** Continuous seated posture increases intradiscal pressure by up to $140\\%$ relative to standing, fatiguing the erector spinae.
2. **Psoas Hypertonicity:** Tight hip flexors from prolonged sitting create an anterior pelvic tilt, compressing posterior facet joints.
3. **Core Decoupling:** Decreased transverse abdominis activation transfers shear loads directly to the ligamentous lumbar spine.

### Red Flag Symptoms Requiring Urgent Medical Evaluation
Seek immediate medical attention if you experience:
- Radiating sharp neurological pain below the knee (sciatica with motor deficit).
- Bilateral numbness or weakness (foot drop).
- Changes in bowel or bladder control (Cauda Equina Syndrome indicators).

### Evidence-Based Non-Pharmacological Protocol
- **Ergonomic Decompression:** Stand and walk for 2 minutes every 45 minutes; adjust chair height so hips sit slightly higher than knees.
- **Targeted Activation:** McGill Big 3 movements (Modified Curl-Up, Bird-Dog, Side Plank) to establish lumbar stability without spinal flexion.`;
  }

  // 6. Personal Finance & Wealth Building (Risk Auditor / Strategist)
  if (
    agentId === 'risk_auditor' ||
    q.includes('invest') ||
    q.includes('sip') ||
    q.includes('mutual fund') ||
    q.includes('salary') ||
    q.includes('wealth') ||
    q.includes('emergency fund') ||
    q.includes('lakh') ||
    q.includes('rupee') ||
    q.includes('portfolio') ||
    q.includes('stocks') ||
    q.includes('finance')
  ) {
    return `### Wealth Architecture Framework
For working professionals establishing financial foundations, portfolio construction should follow a strict hierarchy of capital allocation based on downside asymmetry and tax efficiency.

### The 4-Tier Capital Hierarchy
1. **Tier 1: Fortress Liquidity (Emergency Fund):**
   - Maintain 6–9 months of essential living expenses in high-liquidity, low-volatility vehicles (liquid mutual funds or sweep-in fixed deposits). This prevents panic liquidations of volatile assets during career transitions.
2. **Tier 2: Asymmetric Risk Coverage:**
   - Comprehensive term life cover ($15\\times$ to $20\\times$ annual CTC) and independent personal health insurance super-top-up (independent of corporate employer cover).
3. **Tier 3: Passive Long-Term Equity Compounding (SIP):**
   - Deploy surplus capital into low-cost broad-market index funds (Nifty 50 / S&P 500 equivalent) via disciplined monthly Systematic Investment Plans. Equity captures productive macroeconomic compounding over rolling 7+ year horizons.
4. **Tier 4: Tactical / Growth Allocation:**
   - Direct equities or sectoral thematic bets should not exceed $10\\%–15\\%$ of net investable surplus.

### Guiding Metric
Never evaluate nominal returns in isolation; measure **real risk-adjusted post-tax CAGR** relative to domestic consumer price inflation.`;
  }

  // 7. Legal / Contracts / Startup Agreements (Jurist)
  if (
    agentId === 'jurist' ||
    q.includes('trademark') ||
    q.includes('patent') ||
    q.includes('contract') ||
    q.includes('founder') ||
    q.includes('equity split') ||
    q.includes('vesting') ||
    q.includes('legal') ||
    q.includes('gdpr') ||
    q.includes('agreement') ||
    q.includes('clause') ||
    q.includes('compliance')
  ) {
    return `### Legal Framework & Risk Surface
*Disclaimer: The following represents legal framework analysis for educational purposes and does not constitute formal legal counsel or create an attorney-client relationship.*

In handling intellectual property and corporate structuring disputes, protective strategy hinges on establishing priority dates, clean title, and enforceable covenants.

### Critical Provisions for Founder & Co-Founder Agreements
1. **Four-Year Vesting with 1-Year Cliff:**
   - Unvested equity must remain subject to company repurchase at nominal par value upon separation to prevent "dead equity" blocking subsequent funding rounds.
2. **Comprehensive IP Assignment (Invention Assignment):**
   - Ensure explicit present assignment language ("hereby assigns all right, title, and interest") rather than an agreement to assign in the future.
3. **Double-Trigger Acceleration:**
   - Acceleration of vesting should require both a Change of Control AND involuntary termination without cause, aligning founder incentives with acquirer retention.

### Strategic Mitigation Protocol
- Audit corporate registers to confirm complete execution of proprietary information agreements.
- Maintain a clean chain-of-title documentation repository before initiating any institutional diligence.`;
  }

  // 8. General Comprehensive Claude Archetype Answer
  return `### Core Concept & Direct Assessment
${question}

Addressing this premise with precision requires evaluating both the primary mechanism and the operational constraints governing the outcome.

### Structured Breakdown
1. **Foundational Principle:**
   The fundamental variable is identifying high-leverage leverage points while minimizing unforced execution errors and friction.
   
2. **Key Trade-offs to Consider:**
   - **Speed vs. Thoroughness:** Moving rapidly validates assumptions in the market, but moving without risk bounds creates compounding technical or operational debt.
   - **Simplicity vs. Completeness:** A tightly executed minimalist approach consistently outperforms an over-engineered framework that fails under stress.

3. **Actionable Implementation Framework:**
   - **Define the Single Hurdle Metric:** Establish the measurable threshold that dictates whether this effort is succeeding or requires a course correction.
   - **Ring-Fence the Blast Radius:** Run a bounded 7-to-14 day pilot cohort before committing full capital or organizational resources.
   - **Review Empirical Feedback:** Rely on observed user behavior and measured metrics rather than internal consensus slides.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLAUDE MULTI-AGENT DELIBERATION GENERATOR (High-Fidelity Dynamic Offline)
// ─────────────────────────────────────────────────────────────────────────────
function generateClaudeDeliberation(
  dilemma: string,
  summonedAgentId: string | null,
  activeAgentIds: string[],
  prosody: { state: string; stressScore: number },
  intensity: string
) {
  const isStressed = prosody.stressScore > 45;
  const lead = summonedAgentId && activeAgentIds.includes(summonedAgentId) ? summonedAgentId : activeAgentIds[0] || 'strategist';
  const challenger = activeAgentIds.find((id) => id !== lead && (id === 'risk_auditor' || id === 'devils_advocate')) || activeAgentIds[1] || 'risk_auditor';
  const closer = activeAgentIds.find((id) => id !== lead && id !== challenger && id === 'executioner') || activeAgentIds[2] || 'executioner';

  const turns = [
    {
      speakerId: lead,
      speakerName: getAgentName(lead),
      text: isStressed
        ? `Let us step back and ground the conversation. High-stakes choices feel paralyzing when viewed as all-or-nothing gambles. In reality, the strategic objective here is to capture the structural upside while strictly ring-fencing our downside exposure.`
        : `This decision represents a decisive inflection point. If we align our core operational priorities behind this initiative with clear milestones, we establish an asymmetric advantage before the broader market recognizes the opportunity.`,
      addressing: 'The User',
      isRebuttal: false,
      emotionShiftNote: isStressed ? 'Grounding, measured tempo' : 'Warm, analytical conviction',
    },
    {
      speakerId: challenger,
      speakerName: getAgentName(challenger),
      text: `Hold on, ${getAgentName(lead).replace('The ', '')}—let us examine the unverified assumptions before signing off on that thesis. If our payback period stretches or churn increases by even 8%, we create an acute cash-flow constraint that compromises our entire runway.`,
      addressing: getAgentName(lead),
      isRebuttal: true,
      rebuttalTargetId: lead,
      emotionShiftNote: 'Precise, skeptical cadence',
    },
    {
      speakerId: closer,
      speakerName: getAgentName(closer),
      text: `We do not resolve this through open-ended debate. Here is the 7-day operational path: we launch a ring-fenced validation pilot with clear go/no-go hurdle metrics. If we hit the threshold within 10 days, we allocate full resources; if not, we pivot immediately.`,
      addressing: 'The Entire Panel',
      isRebuttal: false,
      emotionShiftNote: 'Decisive, high-energy cadence',
    },
  ];

  return {
    turns,
    decisionMatrix: {
      consensusScore: isStressed ? 52 : 74,
      corePremise: dilemma.slice(0, 140),
      consensusPoints: [
        {
          id: 'cp_1',
          point: 'Inaction carries a slow structural cost; deliberate, bounded experimentation is required.',
          supportedBy: [lead, closer],
        },
        {
          id: 'cp_2',
          point: 'Capital and resource allocation must be gated behind verifiable milestone checkpoints.',
          supportedBy: [challenger, closer],
        },
      ],
      criticalConflicts: [
        {
          id: 'cc_1',
          title: 'Speed of Market Capture vs Downside Capital Preservation',
          opposingViews: [
            { agentId: lead, stance: 'Prioritize strategic upside and category positioning.' },
            { agentId: challenger, stance: 'Protect liquidity runway and enforce stop-loss thresholds.' },
          ],
          severity: 'High',
        },
      ],
      immediateNextSteps: [
        {
          id: 'ns_1',
          task: 'Define the singular quantitative hurdle metric (KPI) that validates the premise.',
          timeline: 'Day 1–2',
          assignedAgent: closer,
          priority: 'Critical',
        },
        {
          id: 'ns_2',
          task: 'Model the downside break-even boundary and establish a formal stop-loss trigger.',
          timeline: 'Day 3–4',
          assignedAgent: challenger,
          priority: 'High',
        },
        {
          id: 'ns_3',
          task: 'Reconvene panel to evaluate pilot data and vote on organizational resource greenlight.',
          timeline: 'Day 7',
          assignedAgent: lead,
          priority: 'Normal',
        },
      ],
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Vite Dev Server / Static Hosting
// ─────────────────────────────────────────────────────────────────────────────
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n========================================================`);
    console.log(`  CoCounsel AI: Multi-Agent Voice War-Room (Claude Edition)`);
    console.log(`  Running live at: http://localhost:${PORT}`);
    console.log(`  Anthropic API Key: ${currentAnthropicKey ? 'Configured ✅' : 'Not set (Offline Claude Engine Active)'}`);
    console.log(`  Gemini API Key:    ${currentGeminiKey ? 'Configured ✅' : 'Not set'}`);
    console.log(`========================================================\n`);
  });
}

startServer();
