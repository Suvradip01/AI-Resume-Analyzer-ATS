import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Building2, Mail, Lock, Sparkles, CheckCircle2 } from "lucide-react";
import { useClerk } from "@clerk/clerk-react";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE = `${import.meta.env.VITE_API_BASE}/api/v1/recruiter`;

export default function RecruiterLoginPage() {
    const nav = useNavigate();
    const { signOut } = useClerk();
    const [mode, setMode] = useState("login");
    const [company, setCompany] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    async function submit(e) {
        if (e) e.preventDefault();
        setError("");

        if (mode === "register" && !company.trim()) {
            setError("Company name is required.");
            return;
        }
        if (!username.trim() || !password) {
            setError("Email and Password are required.");
            return;
        }

        setLoading(true);
        try {
            const payload = { username: username.trim(), password };
            if (mode === "register") payload.company = company.trim();

            const res = await fetch(`${API_BASE}/${mode}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json().catch(() => ({}));

            if (!res.ok) throw new Error(data.detail || "Authentication failed. Please check your credentials.");

            localStorage.setItem("recruiter_token", data.token);
            localStorage.setItem("recruiter_username", username.trim());
            await signOut();
            nav("/recruiter/dashboard");
        } catch (e) {
            setError(e.message || "Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    }

    const toggleMode = () => {
        setMode(mode === "login" ? "register" : "login");
        setError("");
    };

    return (
        <div className="min-h-screen bg-black text-white flex overflow-hidden font-sans selection:bg-white/20 relative w-full h-full">

            {/* BRANDING PANEL */}
            <motion.div
                initial={false}
                animate={{ x: mode === "login" ? "0%" : "100%" }}
                transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
                style={{ willChange: "transform" }}
                className="hidden lg:flex w-1/2 absolute inset-y-0 left-0 flex-col justify-between p-12 bg-black z-20 border-r border-white/10"
            >
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff03_1px,transparent_1px),linear-gradient(to_bottom,#ffffff03_1px,transparent_1px)] bg-[size:48px_48px]" />
                    {/* High-Performance Gradient instead of Blur Filter */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.03)_0%,transparent_70%)]" />
                </div>

                <div className="relative z-10 flex items-center gap-3">
                    <span className="font-bold text-2xl tracking-tight">InSightATS</span>
                </div>

                <div className="relative z-10 max-w-lg mb-20">
                    <h1 className="text-5xl lg:text-6xl font-semibold tracking-tight mb-8 leading-[1.1] text-white">
                        Identify the top <span className="text-white font-bold drop-shadow-[0_0_15px_rgba(255,255,255,0.4)]">1% of talent</span> in seconds.
                    </h1>
                    <div className="space-y-5">
                        {[
                            "Advanced Semantic Match AI",
                            "Automated Score Breakdowns",
                            "Instant Applicant Shortlisting"
                        ].map((feat, i) => (
                            <div key={i} className="flex items-center gap-4 text-neutral-400">
                                <CheckCircle2 className="size-6 text-white shrink-0" />
                                <span className="text-lg font-medium">{feat}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </motion.div>

            {/* FORM PANEL */}
            <motion.div
                initial={false}
                animate={{ x: mode === "login" ? "100%" : "0%" }}
                transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
                style={{ willChange: "transform" }}
                className="w-full lg:w-1/2 absolute lg:relative inset-0 flex flex-col justify-center p-8 sm:p-12 bg-[#050505] z-10"
            >
                {/* Mobile Header */}
                <div className="absolute top-8 left-8 lg:hidden flex items-center gap-2">
                    <span className="font-bold text-xl tracking-tight">InSightATS</span>
                </div>

                <div className="w-full max-w-[360px] mx-auto">
                    <div className="mb-10">
                        <h2 className="text-[32px] font-bold tracking-tight mb-2 text-white">
                            {mode === "login" ? "Welcome back" : "Create an account"}
                        </h2>
                        <p className="text-neutral-400 text-sm font-medium">
                            {mode === "login"
                                ? "Enter your credentials to access the recruiter dashboard."
                                : "Enter your company details to get started."}
                        </p>
                    </div>

                    <form onSubmit={submit} className="space-y-5">
                        <AnimatePresence mode="wait">
                            {mode === "register" && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.98 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.98 }}
                                    transition={{ duration: 0.1 }}
                                    className="space-y-1.5"
                                >
                                    <label className="text-[13px] font-semibold text-neutral-300">Company Name</label>
                                    <div className="relative group">
                                        <Building2 className="absolute left-3.5 top-1/2 -translate-y-1/2 size-[18px] text-neutral-500 group-focus-within:text-white transition-colors" />
                                        <input
                                            value={company}
                                            onChange={(e) => setCompany(e.target.value)}
                                            placeholder="Acme Corp"
                                            className="w-full pl-[38px] pr-4 py-3 bg-neutral-900 border border-neutral-800 rounded-xl text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:ring-1 focus:ring-white focus:border-white transition-all shadow-sm"
                                        />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <div className="space-y-1.5">
                            <label className="text-[13px] font-semibold text-neutral-300">Work Email / Username</label>
                            <div className="relative group">
                                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 size-[18px] text-neutral-500 group-focus-within:text-white transition-colors" />
                                <input
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    placeholder="name@company.com"
                                    className="w-full pl-[38px] pr-4 py-3 bg-neutral-900 border border-neutral-800 rounded-xl text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:ring-1 focus:ring-white focus:border-white transition-all shadow-sm"
                                />
                            </div>
                        </div>

                        <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                                <label className="text-[13px] font-semibold text-neutral-300">Password</label>
                                {mode === "login" && (
                                    <a href="#" className="text-[13px] font-medium text-neutral-400 hover:text-white transition-colors">Forgot password?</a>
                                )}
                            </div>
                            <div className="relative group">
                                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 size-[18px] text-neutral-500 group-focus-within:text-white transition-colors" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className="w-full pl-[38px] pr-4 py-3 bg-neutral-900 border border-neutral-800 rounded-xl text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:ring-1 focus:ring-white focus:border-white transition-all shadow-sm"
                                />
                            </div>
                        </div>

                        <AnimatePresence>
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                    className="p-3 text-[13px] font-medium text-black bg-white rounded-xl shadow-[0_0_15px_rgba(255,255,255,0.3)]"
                                >
                                    {error}
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full h-12 mt-4 bg-white text-black text-sm font-bold rounded-xl hover:bg-neutral-200 focus:outline-none focus:ring-2 focus:ring-white/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(255,255,255,0.1)] hover:shadow-[0_0_20px_rgba(255,255,255,0.25)]"
                        >
                            {loading ? (
                                <div className="size-4 border-2 border-black/20 border-t-black rounded-full animate-spin" />
                            ) : (
                                mode === "login" ? "Sign In" : "Create Account"
                            )}
                        </button>
                    </form>

                    <div className="mt-8 text-center">
                        <p className="text-[13px] font-medium text-neutral-500">
                            {mode === "login" ? "Don't have an account? " : "Already have an account? "}
                            <button
                                type="button"
                                onClick={toggleMode}
                                className="text-white hover:text-neutral-300 font-bold underline underline-offset-4 transition-colors"
                            >
                                {mode === "login" ? "Sign up" : "Log in"}
                            </button>
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
