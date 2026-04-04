// bad JS: var, ==, eval, console.log, empty catch
var name = "world";
if (name == "world") {            // loose equality
    console.log("Hello " + name);
    eval("alert('xss')");         // eval
    try {
        throw new Error("oops");
    } catch (error) {             // syntax: missing param
    }
}