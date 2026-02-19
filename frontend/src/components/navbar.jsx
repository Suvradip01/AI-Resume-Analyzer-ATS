import { useState } from "react";
import { MenuIcon, XIcon, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { SignedIn, SignedOut, UserButton } from "@clerk/clerk-react";

export default function Navbar() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const navlinks = [
        {
            href: "#creations",
            text: "Creations",
        },
        {
            href: "#about",
            text: "About",
        },
        {
            href: "#contact",
            text: "Contact",
        },
    ];
    return (
        <>
            <motion.nav className="sticky top-0 z-50 flex items-center justify-between w-full h-18 px-6 md:px-16 lg:px-24 xl:px-32 backdrop-blur"
                initial={{ y: -100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 250, damping: 70, mass: 1 }}
            >
                <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition cursor-pointer">
                    <div className="size-8 bg-white rounded-lg flex items-center justify-center shadow-lg shadow-white/20">
                        <Sparkles className="text-black size-5" />
                    </div>
                    <span className="font-bold text-lg tracking-tight text-white">InSightATS</span>
                </Link>

                <div className="hidden lg:flex items-center gap-8 transition duration-500">
                    {navlinks.map((link) => (
                        <Link key={link.href} to={link.href} className="hover:text-slate-300 transition">
                            {link.text}
                        </Link>
                    ))}
                </div>

                <div className="hidden lg:flex items-center space-x-3">
                    <Link to="/dashboard" className="px-6 py-2 bg-primary hover:bg-primary-mid transition text-black rounded-md active:scale-95">
                        Dashboard
                    </Link>
                    <SignedOut>
                        <Link to="/sign-in" className="hover:bg-primary/10 text-primary border border-primary transition px-6 py-2 rounded-md active:scale-95">
                            Sign In
                        </Link>
                    </SignedOut>
                    <SignedIn>
                        <UserButton afterSignOutUrl="/" />
                    </SignedIn>
                </div>
                <button onClick={() => setIsMenuOpen(true)} className="lg:hidden active:scale-90 transition">
                    <MenuIcon className="size-6.5" />
                </button>
            </motion.nav>
            <div className={`fixed inset-0 z-[100] bg-background/80 backdrop-blur flex flex-col items-center justify-center text-lg gap-8 lg:hidden transition-transform duration-400 ${isMenuOpen ? "translate-x-0" : "-translate-x-full"}`}>
                {navlinks.map((link) => (
                    <Link key={link.href} to={link.href} onClick={() => setIsMenuOpen(false)}>
                        {link.text}
                    </Link>
                ))}
                <Link to="/dashboard" onClick={() => setIsMenuOpen(false)}>Dashboard</Link>
                <SignedOut>
                    <Link to="/sign-in" onClick={() => setIsMenuOpen(false)}>Sign In</Link>
                </SignedOut>
                <SignedIn>
                    <UserButton afterSignOutUrl="/" />
                </SignedIn>
                <button onClick={() => setIsMenuOpen(false)} className="active:ring-3 active:ring-white aspect-square size-10 p-1 items-center justify-center bg-slate-100 hover:bg-slate-200 transition text-black rounded-md flex">
                    <XIcon />
                </button>
            </div>
        </>
    );
}