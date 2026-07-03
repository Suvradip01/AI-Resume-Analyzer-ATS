import { UserButton, useUser } from "@clerk/clerk-react";
import { useDispatch, useSelector } from "react-redux";
import { useEffect } from "react";
import {
    Upload, FileText, CheckCircle, AlertCircle, Sparkles,
    ChevronRight, Info, AlertTriangle, CheckCircle2,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import ResultPieChart from "../components/ResultPieChart";
import ResultRadarChart from "../components/ResultRadarChart";
import { Card, CardContent } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";

import {
    setFile,
    setJobDescription,
    runAnalysis,
    clearError,
    selectFile,
    selectJobDescription,
    selectIsLoading,
    selectResult,
    selectError,
    selectAnalysisStatus,
} from "../store/analysisSlice";

const MotionDiv = motion.div;

// ── Pure helpers (no state — memoisation-friendly) ───────────────────────────

function cleanFeedbackText(text) {
    if (!text) return "";
    let cleaned = text.replace(/[\uFFFD\u203D◆♦]/g, "");

    const levelMatch = cleaned.match(/\((Basic|Medium|Advanced)\s*Level\)/i);
    if (levelMatch) {
        cleaned = cleaned.replace(/^.*?(\((Basic|Medium|Advanced)\s*Level\)).*?:\s*/i, "$1. ");
    } else {
        const labelPattern =
            /^[\s\W]*(\*\*)?(Skills|Experience Relevance|Experience|Project Detected|Project|Overall|Tip for [^:]+|Final Verdict)(\*\*)?:\s*/i;
        cleaned = cleaned.replace(labelPattern, "");
    }

    cleaned = cleaned.replace(/^[^a-zA-Z0-9*(\"'`[\]]+/g, "");
    cleaned = cleaned.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>');
    return cleaned.trim();
}

function parseFeedback(feedbackItem) {
    if (feedbackItem.includes("✅ **Skills**"))
        return { type: "skills", status: "success", title: "Skills Match", icon: CheckCircle2 };
    if (feedbackItem.includes("❌ **Critical Missing Skills**"))
        return { type: "skills", status: "warning", title: "Skills Match", icon: AlertTriangle };
    if (feedbackItem.includes("Basic Level") || feedbackItem.includes("Medium Level"))
        return { type: "project", status: "info", title: "Project Complexity", icon: Info };
    if (feedbackItem.includes("Tip for Project"))
        return { type: "project", subtype: "tip", status: "info", title: "Project Recommendations", icon: Info };
    if (feedbackItem.includes("Advanced Level"))
        return { type: "project", status: "success", title: "Project Portfolio", icon: CheckCircle2 };
    if (feedbackItem.includes("Experience Relevance"))
        return { type: "experience", status: "warning", title: "Experience Relevance", icon: AlertTriangle };
    if (feedbackItem.includes("Overall"))
        return { type: "overall", status: "default", title: "Final Verdict", icon: Sparkles };
    return { type: "general", status: "info", title: "Insight", icon: Info };
}

function getCategorizedFeedback(result) {
    if (!result?.feedback) return { skills: [], projects: [], experience: [], other: [] };
    const categories = { skills: [], projects: [], experience: [], other: [] };
    result.feedback.forEach((item) => {
        const parsed = parseFeedback(item);
        const data = { ...parsed, original: item };
        if (parsed.type === "skills")      categories.skills.push(data);
        else if (parsed.type === "project") categories.projects.push(data);
        else if (parsed.type === "experience") categories.experience.push(data);
        else                               categories.other.push(data);
    });
    return categories;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function Dashboard() {
    useUser();
    const dispatch = useDispatch();

    // Redux state
    const file           = useSelector(selectFile);
    const jobDescription = useSelector(selectJobDescription);
    const loading        = useSelector(selectIsLoading);
    const result         = useSelector(selectResult);
    const error          = useSelector(selectError);
    const status         = useSelector(selectAnalysisStatus);

    // Scroll to results when analysis completes
    useEffect(() => {
        if (status === "success") {
            setTimeout(() => {
                document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth" });
            }, 100);
        }
    }, [status]);

    function handleFileChange(e) {
        const selected = e.target.files[0];
        if (!selected) return;
        const allowed = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ];
        if (allowed.includes(selected.type)) {
            dispatch(setFile(selected));
        } else {
            dispatch(setFile(null));
        }
    }

    function handleAnalyze() {
        if (!file || !jobDescription.trim()) return;
        dispatch(runAnalysis({ file, jobDescription }));
    }

    const { skills, projects, experience, other } = getCategorizedFeedback(result);

    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30 overflow-x-hidden">
            {/* Background Gradients & Grid */}
            <div className="fixed inset-0 z-0 pointer-events-none bg-black">
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
                <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-white opacity-20 blur-[100px]" />
            </div>

            {/* Header / Floating Navbar */}
            <header className="fixed top-6 inset-x-0 z-50 flex justify-center px-6">
                <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 flex items-center justify-between shadow-2xl w-full max-w-6xl">
                    <a href="/" className="flex items-center hover:opacity-80 transition cursor-pointer pl-2">
                        <img src="/logo.png" alt="InSightATS Logo" className="h-10 md:h-12 w-auto object-contain scale-[2] md:scale-[2.5] origin-left" />
                    </a>

                    <div className="flex items-center gap-6">
                        <div className="hidden md:flex items-center gap-3 px-4 py-2 bg-white/5 rounded-full border border-white/5 backdrop-blur-sm">
                            <span className="text-xs font-medium text-slate-400">System Operational</span>
                            <div className="flex gap-1.5">
                                {["red", "orange", "green"].map((color, i) => (
                                    <span key={color} className="relative flex h-2 w-2">
                                        <span className={`animate-[ping_1.5s_linear_infinite] ${i > 0 ? `delay-${i * 500}` : ""} absolute inline-flex h-full w-full rounded-full bg-${color}-400 opacity-75`} />
                                        <span className={`relative inline-flex rounded-full h-2 w-2 bg-${color}-500`} />
                                    </span>
                                ))}
                            </div>
                        </div>
                        <UserButton afterSignOutUrl="/" appearance={{ elements: { avatarBox: "size-9 border-2 border-white/10 ring-2 ring-white/5 hover:ring-white/20 transition duration-300" } }} />
                    </div>
                </div>
            </header>

            <main className="relative z-10 w-full max-w-6xl mx-auto px-6 pt-28 pb-10 flex flex-col items-center justify-center min-h-screen">

                {/* Intro */}
                <MotionDiv initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
                    <h1 className="text-3xl md:text-5xl font-bold mb-3 bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-transparent">
                        Optimize Your Resume
                    </h1>
                    <p className="text-neutral-400 text-sm md:text-base max-w-lg mx-auto">
                        Get instant, AI-powered compatibility scoring and feedback.
                    </p>
                </MotionDiv>

                {/* Input Grid */}
                <MotionDiv
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1, duration: 0.4 }}
                    className="w-full grid grid-cols-1 lg:grid-cols-2 gap-6"
                >
                    {/* Resume Upload */}
                    <div className="group relative overflow-hidden rounded-3xl border border-white/10 bg-neutral-900/50 p-6 hover:bg-neutral-900/80 transition duration-500">
                        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500" />

                        <div className="flex items-center justify-between mb-4 relative z-10">
                            <label className="text-xs font-bold text-white uppercase tracking-wider">Resume File</label>
                            {file && <span className="text-xs font-bold text-white bg-green-500 px-3 py-1 rounded-full uppercase tracking-wider shadow-lg shadow-green-500/20">Uploaded</span>}
                        </div>

                        <div className="relative">
                            <input type="file" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20" />
                            <div className={`border border-dashed rounded-2xl h-40 transition-all duration-300 flex flex-col items-center justify-center gap-3 ${file ? "border-white/40 bg-white/5" : "border-neutral-700 hover:border-white/50 hover:bg-white/5"}`}>
                                {file ? (
                                    <>
                                        <div className="size-10 bg-green-500 rounded-xl flex items-center justify-center shadow-lg shadow-green-500/20">
                                            <CheckCircle className="size-5 text-white" />
                                        </div>
                                        <div className="text-center">
                                            <p className="font-medium text-white text-sm truncate max-w-[200px]">{file.name}</p>
                                            <p className="text-xs text-neutral-400">Ready for analysis</p>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <div className="size-10 rounded-xl bg-neutral-800 flex items-center justify-center text-neutral-400 group-hover:bg-white group-hover:text-black transition duration-300">
                                            <Upload className="size-5" />
                                        </div>
                                        <div className="text-center">
                                            <p className="text-sm font-medium text-neutral-300">Click to upload</p>
                                            <p className="text-[10px] text-neutral-500 uppercase tracking-widest mt-1">PDF • DOCX • TXT</p>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Job Description */}
                    <div className="group relative flex-1 overflow-hidden rounded-3xl border border-white/10 bg-neutral-900/50 p-6 hover:bg-neutral-900/80 transition duration-500">
                        <div className="absolute inset-0 bg-gradient-to-bl from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500" />

                        <div className="flex items-center justify-between mb-4 relative z-10">
                            <label className="text-xs font-bold text-white uppercase tracking-wider">Job Description</label>
                            <span className="text-[10px] text-neutral-500">{jobDescription.length} chars</span>
                        </div>

                        <textarea
                            value={jobDescription}
                            onChange={(e) => dispatch(setJobDescription(e.target.value))}
                            placeholder="Paste the job description here..."
                            className="relative z-20 w-full h-[calc(100%-3rem)] bg-transparent border-none text-neutral-200 focus:ring-0 resize-none placeholder:text-neutral-600 text-sm leading-relaxed p-0 scrollbar-hide"
                        />
                    </div>
                </MotionDiv>

                {/* Analyze Button */}
                <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="w-full max-w-md mt-8">
                    <AnimatePresence>
                        {error && (
                            <MotionDiv
                                initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                                animate={{ opacity: 1, height: "auto", marginBottom: 12 }}
                                exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="bg-red-500/10 border border-red-500/20 text-red-200 px-4 py-2 rounded-lg flex items-center gap-2 text-sm justify-center">
                                    <AlertCircle className="size-4 shrink-0" />
                                    <span>{error}</span>
                                    <button onClick={() => dispatch(clearError())} className="ml-auto text-red-400 hover:text-red-200 transition">✕</button>
                                </div>
                            </MotionDiv>
                        )}
                    </AnimatePresence>

                    <button
                        id="analyze-btn"
                        onClick={handleAnalyze}
                        disabled={loading || !file || !jobDescription.trim()}
                        className={`relative group w-full h-14 rounded-full font-bold text-base shadow-[0_0_40px_-10px_rgba(255,255,255,0.3)] flex items-center justify-center gap-2 overflow-hidden transition-all duration-300 ${
                            loading || !file || !jobDescription.trim()
                                ? "bg-neutral-900 text-neutral-500 cursor-not-allowed border border-white/5"
                                : "bg-white text-black hover:scale-[1.02]"
                        }`}
                    >
                        {loading ? (
                            <>
                                <div className="size-4 border-2 border-neutral-400 border-t-transparent rounded-full animate-spin" />
                                <span className="animate-pulse">Analyzing Resume...</span>
                            </>
                        ) : (
                            <>
                                <span>Run Analysis</span>
                                <ChevronRight className="size-4 group-hover:translate-x-1 transition-transform" />
                            </>
                        )}
                    </button>
                </MotionDiv>

                {/* Results */}
                <AnimatePresence>
                    {result && (
                        <MotionDiv
                            id="results-section"
                            initial={{ opacity: 0, y: 40 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="w-full mt-12 space-y-12 pb-20"
                        >
                            {/* Charts */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="bg-neutral-900/40 backdrop-blur-md border border-white/10 rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden">
                                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">Overall Match Score</h2>
                                    <ResultPieChart score={result.score} />
                                </div>
                                <div className="bg-neutral-900/40 backdrop-blur-md border border-white/10 rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden">
                                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">XAI Match Breakdown</h2>
                                    <ResultRadarChart data={[
                                        { subject: "Skills",     A: result.skill_score,      fullMark: 100 },
                                        { subject: "Experience", A: result.experience_score,  fullMark: 100 },
                                        { subject: "Projects",   A: result.project_score,     fullMark: 100 },
                                        { subject: "Structure",  A: result.structure_score,   fullMark: 100 },
                                    ]} />
                                </div>
                            </div>

                            {/* Detailed Insights */}
                            <div className="space-y-6">
                                <div className="flex flex-col gap-1">
                                    <h2 className="text-3xl font-semibold text-white">Detailed Insights</h2>
                                    <p className="text-sm text-neutral-400">Actionable improvements to increase your match score</p>
                                </div>

                                <div className="grid grid-cols-1 gap-6">
                                    {/* Skills */}
                                    {skills.map((item, i) => (
                                        <Card key={`skill-${i}`} className="bg-zinc-900 border-zinc-800 rounded-2xl shadow-sm hover:border-zinc-700 transition-colors">
                                            <CardContent className="p-6 flex flex-col md:flex-row gap-6 items-start md:items-center justify-between">
                                                <div className="flex items-start gap-4">
                                                    <div className={`p-2 rounded-full mt-1 ${item.status === "success" ? "bg-green-500/10" : "bg-red-500/10"}`}>
                                                        <item.icon className={`size-6 ${item.status === "success" ? "text-green-500" : "text-red-500"}`} />
                                                    </div>
                                                    <div>
                                                        <div className="flex items-center gap-3 mb-1">
                                                            <h3 className="font-semibold text-white text-lg">Skills Match</h3>
                                                            <Badge variant={item.status === "success" ? "success" : "destructive"} className="rounded-full px-3">
                                                                {item.status === "success" ? "Strong" : "Action Needed"}
                                                            </Badge>
                                                        </div>
                                                        <div className="text-slate-300 text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: cleanFeedbackText(item.original) }} />
                                                        {item.status === "success" && <div className="mt-3 w-32"><Progress value={100} className="h-1.5" /></div>}
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}

                                    {/* Projects */}
                                    {projects.map((item, i) => (
                                        <Card key={`proj-${i}`} className="bg-zinc-900 border-zinc-800 rounded-2xl shadow-sm hover:border-zinc-700 transition-colors">
                                            <CardContent className="p-6 flex items-start gap-4">
                                                <div className={`p-2 rounded-full mt-1 ${item.status === "success" ? "bg-green-500/10" : "bg-blue-500/10"}`}>
                                                    <item.icon className={`size-6 ${item.status === "success" ? "text-green-500" : "text-blue-500"}`} />
                                                </div>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-3 mb-1">
                                                        <h3 className="font-semibold text-white text-lg">{item.title}</h3>
                                                        <Badge variant={item.status === "success" ? "success" : "info"} className="rounded-full px-3">
                                                            {item.status === "success" ? "Strong" : "Improve"}
                                                        </Badge>
                                                    </div>
                                                    <div className="text-slate-300 text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: cleanFeedbackText(item.original) }} />
                                                    {item.status === "success" && <div className="mt-3 w-32"><Progress value={100} className="h-1.5" /></div>}
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}

                                    {/* Experience */}
                                    {experience.map((item, i) => (
                                        <Card key={`exp-${i}`} className="bg-zinc-900 border-zinc-800 rounded-2xl shadow-sm hover:border-zinc-700 transition-colors">
                                            <CardContent className="p-6 flex items-start gap-4">
                                                <div className="p-2 rounded-full mt-1 bg-yellow-500/10">
                                                    <AlertTriangle className="size-6 text-yellow-500" />
                                                </div>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-3 mb-1">
                                                        <h3 className="font-semibold text-white text-lg">Experience Relevance</h3>
                                                        <Badge variant="warning" className="rounded-full px-3">Attention</Badge>
                                                    </div>
                                                    <div className="text-slate-300 text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: cleanFeedbackText(item.original) }} />
                                                    {result.missing_skills?.length > 0 && i === 0 && (
                                                        <div className="flex flex-wrap gap-2 mt-3">
                                                            {result.missing_skills.slice(0, 5).map((skill, k) => (
                                                                <Badge key={k} variant="outline" className="text-yellow-500 border-yellow-500/30 bg-yellow-500/5">{skill}</Badge>
                                                            ))}
                                                            {result.missing_skills.length > 5 && (
                                                                <span className="text-xs text-neutral-500 self-center">+{result.missing_skills.length - 5} more</span>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}

                                    {/* Other / Final Verdict */}
                                    {other.map((item, i) => {
                                        const isFinal = item.type === "overall";
                                        return (
                                            <Card key={`other-${i}`} className={`rounded-2xl shadow-sm transition-colors ${isFinal ? "bg-gradient-to-br from-zinc-900 to-blue-900/20 border-blue-500/30 hover:border-blue-500/40" : "bg-zinc-900 border-zinc-800 hover:border-zinc-700"}`}>
                                                <CardContent className="p-6 flex items-start gap-4">
                                                    <div className={`p-2 rounded-full mt-1 ${isFinal ? "bg-blue-500/10" : "bg-neutral-800"}`}>
                                                        <item.icon className={`size-6 ${isFinal ? "text-blue-400" : "text-neutral-400"}`} />
                                                    </div>
                                                    <div className="flex-1">
                                                        <h3 className="font-semibold text-white text-lg mb-1">{item.title}</h3>
                                                        <div className="text-slate-300 text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: cleanFeedbackText(item.original) }} />
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        );
                                    })}
                                </div>
                            </div>
                        </MotionDiv>
                    )}
                </AnimatePresence>
            </main>
        </div>
    );
}
