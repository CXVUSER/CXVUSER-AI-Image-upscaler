#if _WIN32
#include <Windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call,
                      LPVOID lpReserved)// reserved
{
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            break;

        case DLL_THREAD_ATTACH: {
            DisableThreadLibraryCalls(hModule);
        } break;
        case DLL_THREAD_DETACH: {
        } break;
        case DLL_PROCESS_DETACH:
            break;
    }
    return TRUE;
}
#endif