import { useState } from "react";
import { MenuIcon, XIcon, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { SignedIn, SignedOut, UserButton } from "@clerk/clerk-react";

export default function Navbar() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false); // Recruiter profile dropdown
    const [dashDropOpen, setDashDropOpen] = useState(false);
    const [signDropOpen, setSignDropOpen] = useState(false);
    
    const MotionNav = motion.nav;
    const navlinks = [
        { href: "#creations", text: "Creations" },
        { href: "#about", text: "About" },
        { href: "#contact", text: "Contact" },
    ];

    return (
        <>
            <MotionNav className="sticky top-0 z-50 flex items-center justify-between w-full h-18 px-6 md:px-16 lg:px-24 xl:px-32 backdrop-blur"
                initial={{ y: -100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 250, damping: 70, mass: 1 }}
            >
                <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition cursor-pointer">
                    <span className="font-bold text-lg tracking-tight text-white">InSightATS</span>
                </Link>

                <div className="hidden lg:flex items-center gap-8 transition duration-500">
                    {navlinks.map((link) => (
                        <Link key={link.href} to={link.href} className="hover:text-slate-300 transition">
                            {link.text}
                        </Link>
                    ))}
                </div>

                <div className="hidden lg:flex items-center space-x-4">
                    
                    {/* NO ONE SIGNED IN (Neither Recruiter nor Candidate) */}
                    {!localStorage.getItem("recruiter_username") && (
                        <SignedOut>
                            <div className="flex items-center gap-4 relative">
                                <div className="relative">
                                    <button
                                        onClick={() => { setDashDropOpen(!dashDropOpen); setSignDropOpen(false); }}
                                        className="px-6 py-2 bg-primary hover:bg-white transition text-black text-sm font-medium rounded-md active:scale-95"
                                    >
                                        Dashboard
                                    </button>
                                    {dashDropOpen && (
                                        <div className="absolute top-full mt-3 left-0 w-48 bg-neutral-900 border border-white/10 rounded-xl shadow-2xl py-2 z-50 flex flex-col">
                                            <Link to="/dashboard" onClick={() => setDashDropOpen(false)} className="px-4 py-3 hover:bg-white/5 text-sm text-white font-medium text-left transition text-center border-b border-white/5">As Candidate</Link>
                                            <Link to="/recruiter/dashboard" onClick={() => setDashDropOpen(false)} className="px-4 py-3 hover:bg-white/5 text-sm text-white font-medium text-left transition text-center">As Recruiter</Link>
                                        </div>
                                    )}
                                </div>

                                <div className="relative">
                                    <button
                                        onClick={() => { setSignDropOpen(!signDropOpen); setDashDropOpen(false); }}
                                        className="hover:bg-primary/10 text-primary border border-primary transition px-5 py-2 rounded-md active:scale-95 text-sm font-medium"
                                    >
                                        Sign In
                                    </button>
                                    {signDropOpen && (
                                        <div className="absolute top-full mt-3 right-0 w-48 bg-neutral-900 border border-white/10 rounded-xl shadow-2xl py-2 z-50 flex flex-col">
                                            <Link to="/sign-in" onClick={() => setSignDropOpen(false)} className="px-4 py-3 hover:bg-white/5 text-sm text-white font-medium text-left transition text-center border-b border-white/5">As Candidate</Link>
                                            <Link to="/recruiter" onClick={() => setSignDropOpen(false)} className="px-4 py-3 hover:bg-white/5 text-sm text-white font-medium text-left transition text-center">As Recruiter</Link>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </SignedOut>
                    )}

                    {/* ONLY RECRUITER SIGNED IN */}
                    {localStorage.getItem("recruiter_username") && (
                        <div className="flex items-center gap-4">
                            <Link to="/recruiter/dashboard" className="px-6 py-2 bg-primary hover:bg-white transition text-black text-sm font-medium rounded-md active:scale-95">
                                Dashboard
                            </Link>

                            <div className="relative">
                                <button
                                    onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                                    title="Recruiter Auth"
                                    className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900 rounded-full border border-white/20 hover:bg-neutral-800 transition cursor-pointer"
                                >
                                    <div className="size-7 rounded-full bg-primary text-black flex items-center justify-center text-sm font-bold shadow-[0_0_10px_rgba(255,255,255,0.4)]">
                                        {localStorage.getItem("recruiter_username")[0].toUpperCase()}
                                    </div>
                                    <div className="flex flex-col pr-2 text-left">
                                        <span className="text-white text-sm font-medium leading-tight">{localStorage.getItem("recruiter_username")}</span>
                                        <span className="text-[10px] text-neutral-400 uppercase tracking-wider leading-tight">Recruiter</span>
                                    </div>
                                </button>
                                {isDropdownOpen && (
                                    <div className="absolute right-0 mt-2 w-48 bg-neutral-900 border border-white/10 rounded-xl shadow-2xl py-2 z-50">
                                        <button
                                            onClick={() => {
                                                localStorage.removeItem("recruiter_token");
                                                localStorage.removeItem("recruiter_username");
                                                setIsDropdownOpen(false);
                                                window.location.reload();
                                            }}
                                            className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-white/5 transition font-medium"
                                        >
                                            Log out
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* ONLY CANDIDATE SIGNED IN */}
                    {!localStorage.getItem("recruiter_username") && (
                        <SignedIn>
                            <div className="flex items-center gap-4">
                                <Link to="/dashboard" className="px-6 py-2 bg-primary hover:bg-white transition text-black text-sm font-medium rounded-md active:scale-95">
                                    Dashboard
                                </Link>

                                <div title="Candidate Auth" className="flex items-center gap-2 bg-white/5 pr-4 pl-1.5 py-1.5 rounded-full border border-white/10">
                                    <UserButton afterSignOutUrl="/" />
                                    <span className="text-xs uppercase tracking-widest text-neutral-400 font-bold">Candidate</span>
                                </div>
                            </div>
                        </SignedIn>
                    )}

                </div>
                
                <button onClick={() => setIsMenuOpen(true)} className="lg:hidden active:scale-90 transition text-white">
                    <MenuIcon className="size-6.5" />
                </button>
            </MotionNav>


            {/* MOBILE MENU */}
            <div className={`fixed inset-0 z-[100] bg-background/80 backdrop-blur flex flex-col items-center justify-center text-lg gap-8 lg:hidden transition-transform duration-400 ${isMenuOpen ? "translate-x-0" : "-translate-x-full"}`}>
                
                {navlinks.map((link) => (
                    <Link key={link.href} to={link.href} onClick={() => setIsMenuOpen(false)}>
                        {link.text}
                    </Link>
                ))}

                {/* NO ONE SIGNED IN */}
                {!localStorage.getItem("recruiter_username") && (
                    <SignedOut>
                        <div className="flex flex-col items-center gap-4 mt-4 bg-white/5 rounded-2xl w-3/4 py-6 border border-white/10 shadow-xl">
                            <span className="text-xs text-neutral-500 uppercase tracking-widest font-bold">Go to Dashboard</span>
                            <Link to="/dashboard" onClick={() => setIsMenuOpen(false)} className="text-lg font-medium text-white hover:text-primary transition">As Candidate</Link>
                            <Link to="/recruiter/dashboard" onClick={() => setIsMenuOpen(false)} className="text-lg font-medium text-white hover:text-primary transition">As Recruiter</Link>
                            
                            <div className="w-1/2 h-px bg-white/10 my-2"></div>
                            
                            <span className="text-xs text-neutral-500 uppercase tracking-widest font-bold">Sign In</span>
                            <Link to="/sign-in" onClick={() => setIsMenuOpen(false)} className="text-lg font-medium text-primary hover:text-white transition">As Candidate</Link>
                            <Link to="/recruiter" onClick={() => setIsMenuOpen(false)} className="text-lg font-medium text-primary hover:text-white transition">As Recruiter</Link>
                        </div>
                    </SignedOut>
                )}

                {/* RECRUITER SIGNED IN */}
                {localStorage.getItem("recruiter_username") && (
                    <>
                        <Link to="/recruiter/dashboard" onClick={() => setIsMenuOpen(false)} className="px-6 py-2 bg-primary text-black rounded-md font-bold">Dashboard</Link>
                        
                        <div className="flex flex-col items-center gap-4 mt-4">
                            <div className="flex items-center gap-2 px-4 py-2 bg-neutral-900 rounded-full border border-white/20">
                                <div className="size-8 rounded-full bg-primary text-black flex items-center justify-center text-sm font-bold shadow-[0_0_10px_rgba(255,255,255,0.4)]">
                                    {localStorage.getItem("recruiter_username")[0].toUpperCase()}
                                </div>
                                <div className="flex flex-col pr-2 text-left">
                                    <span className="text-white font-medium leading-tight">{localStorage.getItem("recruiter_username")}</span>
                                    <span className="text-[10px] text-neutral-400 uppercase tracking-wider leading-tight">Recruiter</span>
                                </div>
                            </div>
                            <button
                                onClick={() => {
                                    localStorage.removeItem("recruiter_token");
                                    localStorage.removeItem("recruiter_username");
                                    window.location.reload();
                                }}
                                className="text-red-400 text-sm font-medium hover:underline px-4 py-2"
                            >
                                Log out
                            </button>
                        </div>
                    </>
                )}

                {/* CANDIDATE SIGNED IN */}
                {!localStorage.getItem("recruiter_username") && (
                    <SignedIn>
                        <Link to="/dashboard" onClick={() => setIsMenuOpen(false)} className="px-6 py-2 bg-primary text-black rounded-md font-bold">Dashboard</Link>
                        
                        <div className="flex items-center gap-3 mt-4 mb-2 bg-white/5 py-2 px-6 rounded-full border border-white/10">
                            <UserButton afterSignOutUrl="/" />
                            <span className="text-sm uppercase tracking-widest text-neutral-400 font-bold">Candidate</span>
                        </div>
                    </SignedIn>
                )}

                <button onClick={() => setIsMenuOpen(false)} className="mt-4 active:ring-3 active:ring-white aspect-square size-10 p-1 items-center justify-center bg-slate-100 hover:bg-slate-200 transition text-black rounded-md flex">
                    <XIcon />
                </button>
            </div>
        </>
    );
}