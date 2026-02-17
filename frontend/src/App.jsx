import { Routes, Route } from "react-router-dom";
import { SignedIn, SignedOut, RedirectToSignIn } from "@clerk/clerk-react";

import GetInTouch from "./sections/get-in-touch";
import TrustedCompanies from "./sections/trusted-companies";
import Footer from "./components/footer";
import LenisScroll from "./components/lenis-scroll";
import Navbar from "./components/navbar";
import AboutOurApps from "./sections/about-our-apps";
import HeroSection from "./sections/hero-section";
import OurLatestCreation from "./sections/our-latest-creation";

// Pages
import Dashboard from "./pages/Dashboard";
import SignInPage from "./pages/sign-in";

function LandingPage() {
    return (
        <>
            <LenisScroll />
            <Navbar />
            <main className="px-6 md:px-16 lg:px-24 xl:px-32">
                <HeroSection />
                <OurLatestCreation />
                <AboutOurApps />
                {/* <OurTestimonials /> Removed per user request */}
                {/* <TrustedCompanies /> Removed per user request */}
                <GetInTouch />
                {/* <SubscribeNewsletter /> Removed per user request */}
            </main>
            {/* <Footer /> Removed per user request */}
        </>
    );
}

export default function App() {
    return (
        <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/sign-in/*" element={<SignInPage />} />
            <Route
                path="/dashboard"
                element={
                    <>
                        <SignedIn>
                            <Dashboard />
                        </SignedIn>
                        <SignedOut>
                            <RedirectToSignIn />
                        </SignedOut>
                    </>
                }
            />
        </Routes>
    );
}