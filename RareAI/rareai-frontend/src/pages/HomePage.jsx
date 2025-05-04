import React from "react";
import { useNavigate } from "react-router-dom";

const HomePage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen flex flex-col bg-gray-100 w-full">
           
            {/* Navbar - full width */}
            <nav className="w-full bg-white shadow-md p-4">
                <div className="w-full max-w-7xl mx-auto flex justify-between items-center px-4 sm:px-6 lg:px-8">
                    <h1 className="text-xl sm:text-2xl font-bold text-orange-500">RareAI</h1>
                    <div className="text-sm sm:text-base">Menu</div>
                </div>
            </nav>

            {/* Main content - full width with centered content */}
            <main className="flex-grow w-full flex items-center justify-center p-4 sm:p-6">
                <div className="w-full px-4 sm:px-6 lg:px-8"> {/* Full width container with padding */}
                    <div className="mx-auto w-full max-w-4xl"> {/* Centered content with max-width */}
                        <div className="bg-white shadow-lg rounded-2xl p-6 sm:p-8 md:p-10 text-center w-full">
                            <h2 className="text-2xl sm:text-3xl md:text-4xl font-semibold mb-4 text-gray-800">
                                Start Your Checkup
                            </h2>
                            <p className="text-sm sm:text-base md:text-lg text-gray-600 mb-6 md:mb-8">
                                Upload a video or record to begin your health checkup.
                            </p>
                            <button
                                className="bg-orange-500 text-white px-6 py-3 rounded-xl w-full text-base sm:text-lg md:text-xl hover:bg-orange-600 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-opacity-50"
                                onClick={() => navigate("/videoupload")}
                            >
                                Start the Checkup
                            </button>
                        </div>
                    </div>
                </div>
            </main>

            {/* Footer - full width */}
            <footer className="w-full p-4 text-center text-xs sm:text-sm text-gray-500">
                <div className="max-w-7xl mx-auto">
                    © {new Date().getFullYear()} RareAI. All rights reserved.
                </div>
            </footer>
        </div>
    );
};

export default HomePage;