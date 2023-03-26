import MenuRouter from "./MenuRouter";
import {Outlet} from "react-router-dom";

export default function Layout(){
    return(
        <div>
            <MenuRouter/>
            <Outlet/>
        </div>

    )
}