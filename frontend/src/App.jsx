import { Navigate, Routes, Route } from "react-router-dom";
import { SignedIn, SignedOut, RedirectToSignIn } from "@clerk/clerk-react";

import GetInTouch from "./sections/get-in-touch";
import LenisScroll from "./components/lenis-scroll";
import Navbar from "./components/navbar";
import AboutOurApps from "./sections/about-our-apps";
import HeroSection from "./sections/hero-section";
import OurLatestCreation from "./sections/our-latest-creation";

// Pages
import Dashboard from "./pages/Dashboard";
import SignInPage from "./pages/sign-in";
import SignUpPage from "./pages/sign-up";
import RecruiterLoginPage from "./pages/recruiter-login";
import RecruiterDashboardPage from "./pages/recruiter-dashboard";

/** Guards the recruiter dashboard — redirects to login if no valid token in storage. */
function RequireRecruiter({ children }) {
    const token = localStorage.getItem("recruiter_token");
    if (!token) return <Navigate to="/recruiter" replace />;
    return children;
}

function LandingPage() {
    return (
        <>
            <LenisScroll />
            <Navbar />
            <main className="px-6 md:px-16 lg:px-24 xl:px-32">
                <HeroSection />
                <OurLatestCreation />
                <AboutOurApps />
                <GetInTouch />
            </main>
        </>
    );
}

export default function App() {
    return (
        <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/sign-in/*" element={<SignInPage />} />
            <Route path="/sign-up/*" element={<SignUpPage />} />
            <Route path="/recruiter" element={<RecruiterLoginPage />} />
            <Route
                path="/recruiter/dashboard"
                element={
                    <RequireRecruiter>
                        <RecruiterDashboardPage />
                    </RequireRecruiter>
                }
            />
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