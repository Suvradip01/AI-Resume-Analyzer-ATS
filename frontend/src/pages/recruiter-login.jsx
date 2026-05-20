import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Building2, Mail, Lock, CheckCircle2, KeyRound } from "lucide-react";
import { useSignIn, useSignUp, useUser } from "@clerk/clerk-react";
import { motion, AnimatePresence } from "framer-motion";

export default function RecruiterLoginPage() {
    const nav = useNavigate();
    const { isLoaded: isSignInLoaded, signIn, setActive: setSignInActive } = useSignIn();
    const { isLoaded: isSignUpLoaded, signUp, setActive: setSignUpActive } = useSignUp();
    const { isLoaded: isUserLoaded, isSignedIn } = useUser();
    
    const [mode, setMode] = useState("login"); // "login", "register", "verify_signup", "forgot_password", "reset_password"
    const [company, setCompany] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [code, setCode] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        if (isUserLoaded && isSignedIn) {
            nav("/recruiter/dashboard");
        }
    }, [isUserLoaded, isSignedIn, nav]);

    if (!isUserLoaded) {
        return (
            <div className="min-h-screen bg-black flex items-center justify-center">
                <div className="size-8 border-4 border-white/20 border-t-white rounded-full animate-spin"></div>
            </div>
        );
    }

    async function submit(e) {
        if (e) e.preventDefault();
        setError("");

        if (mode === "register" && !company.trim()) {
            setError("Company name is required.");
            return;
        }
        if ((mode === "login" || mode === "register") && (!username.trim() || !password)) {
            setError("Email and Password are required.");
            return;
        }
        if (mode === "reset_password" && !password) {
            setError("New Password is required.");
            return;
        }
        if (mode === "register" || mode === "reset_password") {
            const passwordRegex = /^(?=.*[0-9])(?=.*[^a-zA-Z0-9]).{8,}$/;
            if (!passwordRegex.test(password)) {
                setError("Password must be at least 8 characters long, including a number and a special character.");
                return;
            }
        }
        if (mode === "forgot_password" && !username.trim()) {
            setError("Email is required.");
            return;
        }
        if ((mode === "verify_signup" || mode === "reset_password") && !code.trim()) {
            setError("Verification code is required.");
            return;
        }

        setLoading(true);
        try {
            if (mode === "register") {
                if (!isSignUpLoaded) return;
                await signUp.create({
                    emailAddress: username.trim(),
                    password,
                    unsafeMetadata: { company: company.trim() }
                });
                await signUp.prepareEmailAddressVerification({ strategy: "email_code" });
                setMode("verify_signup");
            } 
            else if (mode === "verify_signup") {
                if (!isSignUpLoaded) return;
                const completeSignUp = await signUp.attemptEmailAddressVerification({ code });
                if (completeSignUp.status === "complete") {
                    await setSignUpActive({ session: completeSignUp.createdSessionId });
                    nav("/recruiter/dashboard");
                }
            }
            else if (mode === "login") {
                if (!isSignInLoaded) return;
                const completeSignIn = await signIn.create({
                    identifier: username.trim(),
                    password,
                });
                if (completeSignIn.status === "complete") {
                    await setSignInActive({ session: completeSignIn.createdSessionId });
                    nav("/recruiter/dashboard");
                }
            }
            else if (mode === "forgot_password") {
                if (!isSignInLoaded) return;
                await signIn.create({
                    strategy: "reset_password_email_code",
                    identifier: username.trim(),
                });
                setMode("reset_password");
            }
            else if (mode === "reset_password") {
                if (!isSignInLoaded) return;
                const result = await signIn.attemptFirstFactor({
                    strategy: "reset_password_email_code",
                    code,
                    password,
                });
                if (result.status === "complete") {
                    await setSignInActive({ session: result.createdSessionId });
                    nav("/recruiter/dashboard");
                }
            }
        } catch (e) {
            setError(e.errors?.[0]?.longMessage || e.message || "Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    }

    const toggleMode = () => {
        setMode(mode === "login" ? "register" : "login");
        setError("");
        setCode("");
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

                <div className="relative z-10 flex items-center pl-4">
                    <img src="/logo.png" alt="InSightATS Logo" className="h-16 lg:h-20 w-auto object-contain scale-[2] lg:scale-[2.5] origin-left" />
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
                <div className="absolute top-8 left-8 lg:hidden flex items-center pl-2">
                    <img src="/logo.png" alt="InSightATS Logo" className="h-10 sm:h-12 w-auto object-contain scale-[2] sm:scale-[2.5] origin-left" />
                </div>

                <div className="w-full max-w-[360px] mx-auto">
                    <div className="mb-10">
                        <h2 className="text-[32px] font-bold tracking-tight mb-2 text-white">
                            {mode === "login" && "Welcome back"}
                            {mode === "register" && "Create an account"}
                            {mode === "verify_signup" && "Verify your email"}
                            {mode === "forgot_password" && "Reset Password"}
                            {mode === "reset_password" && "New Password"}
                        </h2>
                        <p className="text-neutral-400 text-sm font-medium">
                            {mode === "login" && "Enter your credentials to access the recruiter dashboard."}
                            {mode === "register" && "Enter your company details to get started."}
                            {mode === "verify_signup" && `We sent a code to ${username}`}
                            {mode === "forgot_password" && "Enter your email to receive a reset code."}
                            {mode === "reset_password" && "Enter the code sent to your email and your new password."}
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

                        {(mode === "login" || mode === "register" || mode === "forgot_password") && (
                            <div className="space-y-1.5">
                                <label className="text-[13px] font-semibold text-neutral-300">Work Email</label>
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
                        )}

                        {(mode === "verify_signup" || mode === "reset_password") && (
                            <div className="space-y-1.5">
                                <label className="text-[13px] font-semibold text-neutral-300">6-Digit Code</label>
                                <div className="relative group">
                                    <KeyRound className="absolute left-3.5 top-1/2 -translate-y-1/2 size-[18px] text-neutral-500 group-focus-within:text-white transition-colors" />
                                    <input
                                        value={code}
                                        onChange={(e) => setCode(e.target.value)}
                                        placeholder="123456"
                                        className="w-full pl-[38px] pr-4 py-3 bg-neutral-900 border border-neutral-800 rounded-xl text-sm text-white placeholder:text-neutral-600 focus:outline-none focus:ring-1 focus:ring-white focus:border-white transition-all shadow-sm"
                                    />
                                </div>
                            </div>
                        )}

                        {(mode === "login" || mode === "register" || mode === "reset_password") && (
                            <div className="space-y-1.5">
                                <div className="flex items-center justify-between">
                                    <label className="text-[13px] font-semibold text-neutral-300">
                                        {mode === "reset_password" ? "New Password" : "Password"}
                                    </label>
                                    {mode === "login" && (
                                        <button type="button" onClick={() => { setMode("forgot_password"); setError(""); }} className="text-[13px] font-medium text-neutral-400 hover:text-white transition-colors">Forgot password?</button>
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
                        )}

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
                                mode === "login" ? "Sign In" : 
                                mode === "register" ? "Create Account" : 
                                mode === "verify_signup" ? "Verify Code" :
                                mode === "forgot_password" ? "Send Reset Code" :
                                "Reset Password"
                            )}
                        </button>
                    </form>

                    <div className="mt-8 text-center">
                        <p className="text-[13px] font-medium text-neutral-500">
                            {(mode === "login" || mode === "forgot_password") ? "Don't have an account? " : "Already have an account? "}
                            <button
                                type="button"
                                onClick={toggleMode}
                                className="text-white hover:text-neutral-300 font-bold underline underline-offset-4 transition-colors"
                            >
                                {(mode === "login" || mode === "forgot_password") ? "Sign up" : "Log in"}
                            </button>
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
