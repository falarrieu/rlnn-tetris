import './css/App.css';
import {BrowserRouter, Route, Routes} from "react-router-dom";
import Charts from './charts/Charts';
import Home from './home/Home';
import Tetris from './tetris/Tetris';
import Layout from './Layout';

export default function App() {

    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Layout/>}>
                    <Route path="/Home" element={<Home/>}/>
                    <Route path="/Charts" element={<Charts/>}/>
                    <Route path="/Tetris" element={<Tetris/>}/>
                </Route>
            </Routes>
        </BrowserRouter>
    );
}