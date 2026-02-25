#include "OverlayUI.h"
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <algorithm>

CyberpunkUI::CyberpunkUI()
{
    mRewardHistory.Clear();
}

void CyberpunkUI::Init(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    ApplyCyberpunkStyle();
}

void CyberpunkUI::ApplyCyberpunkStyle()
{
    ImGuiStyle& style = ImGui::GetStyle();
    
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(6, 4);
    style.CellPadding = ImVec2(4, 2);
    style.ItemSpacing = ImVec2(8, 4);
    style.ItemInnerSpacing = ImVec2(6, 4);
    style.TouchExtraPadding = ImVec2(0, 0);
    style.IndentSpacing = 20.0f;
    style.ScrollbarSize = 14.0f;
    style.GrabMinSize = 10.0f;
    
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;
    
    style.WindowRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.PopupRounding = 3.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_Left;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);
    
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = mColorText;
    colors[ImGuiCol_TextDisabled] = mColorDim;
    colors[ImGuiCol_WindowBg] = ImVec4(0.03f, 0.03f, 0.07f, 0.95f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.02f, 0.02f, 0.05f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.04f, 0.04f, 0.09f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.0f, 1.0f, 0.85f, 0.3f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.1f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.15f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.25f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.0f, 0.5f, 0.4f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.0f, 1.0f, 0.85f, 1.0f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0f, 0.3f, 0.25f, 0.8f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.04f, 0.04f, 0.08f, 1.0f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.05f, 0.6f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.0f, 1.0f, 0.85f, 0.4f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.6f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.8f);
    colors[ImGuiCol_CheckMark] = mColorAccent;
    colors[ImGuiCol_SliderGrab] = mColorAccent;
    colors[ImGuiCol_SliderGrabActive] = mColorAccent2;
    colors[ImGuiCol_Button] = ImVec4(0.0f, 1.0f, 0.85f, 0.2f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.4f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.6f);
    colors[ImGuiCol_Header] = ImVec4(0.0f, 1.0f, 0.85f, 0.2f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.35f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.5f);
    colors[ImGuiCol_Separator] = ImVec4(0.0f, 1.0f, 0.85f, 0.25f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.0f, 1.0f, 0.85f, 0.5f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.0f, 1.0f, 0.85f, 0.7f);
    colors[ImGuiCol_ResizeGrip] = mColorAccent;
    colors[ImGuiCol_ResizeGripHovered] = mColorAccent;
    colors[ImGuiCol_ResizeGripActive] = mColorAccent2;
    colors[ImGuiCol_Tab] = ImVec4(0.0f, 0.5f, 0.4f, 0.8f);
    colors[ImGuiCol_TabHovered] = mColorAccent;
    colors[ImGuiCol_TabActive] = ImVec4(0.0f, 0.8f, 0.7f, 1.0f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.0f, 0.3f, 0.25f, 0.8f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.0f, 0.5f, 0.4f, 1.0f);
    colors[ImGuiCol_PlotLines] = mColorAccent;
    colors[ImGuiCol_PlotLinesHovered] = mColorAccent2;
    colors[ImGuiCol_PlotHistogram] = mColorAccent;
    colors[ImGuiCol_PlotHistogramHovered] = mColorAccent2;
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.0f, 1.0f, 0.85f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = mColorWarning;
    colors[ImGuiCol_NavHighlight] = mColorAccent;
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.0f, 1.0f, 0.85f, 0.6f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.8f, 0.8f, 0.8f, 0.2f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.6f);
}

void CyberpunkUI::BeginFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void CyberpunkUI::EndFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void CyberpunkUI::Shutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void CyberpunkUI::SetTrainingStats(int totalSteps, int episodes, float sps, float avgReward)
{
    mTotalSteps = totalSteps;
    mEpisodes = episodes;
    mSPS = sps;
    mAvgReward = avgReward;
}

void CyberpunkUI::SetEnvironmentStats(int numEnvs, int selectedEnv)
{
    mNumEnvs = numEnvs;
    mSelectedEnv = selectedEnv;
}

void CyberpunkUI::PushRewardHistory(float damageDealt, float damageTaken, float airtime, float energy, float scalar)
{
    mRewardHistory.Push(damageDealt, damageTaken, airtime, energy, scalar);
}

void CyberpunkUI::DrawMainWindow()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 280), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("JOLTrl HYBRID OVERSEER", nullptr, ImGuiWindowFlags_NoCollapse);
    
    ImGui::TextColored(mColorAccent, "SILICON: Intel i5-10500");
    ImGui::TextColored(mColorDim, "GRAPHICS: Intel HD 630");
    ImGui::Separator();
    
    ImGui::Text("PARALLEL ENVS: %d", mNumEnvs);
    ImGui::Text("TOTAL STEPS: %d", mTotalSteps);
    ImGui::TextColored(mColorWarning, "SPS: %.0f", mSPS);
    ImGui::Text("EPISODES: %d", mEpisodes);
    ImGui::Text("AVG REWARD: %.3f", mAvgReward);
    ImGui::Separator();
    
    ImGui::Checkbox("ENABLE RENDERING", &mRenderEnabled);
    ImGui::SliderInt("WATCH ENV", &mSelectedEnv, 0, mNumEnvs - 1);
    
    if (ImGui::Button(mPaused ? "RESUME" : "PAUSE", ImVec2(100, 25)))
    {
        mPaused = !mPaused;
    }
    
    ImGui::End();
}

void CyberpunkUI::DrawRewardGraphs()
{
    ImGui::SetNextWindowPos(ImVec2(10, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 320), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("MOMA-TD3 REWARD STREAMS", nullptr, ImGuiWindowFlags_NoCollapse);
    
    if (mRewardHistory.count > 0)
    {
        int displayCount = mRewardHistory.count;
        
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "DAMAGE DEALT");
        ImGui::PlotLines("##damage_dealt", 
            mRewardHistory.damageDealtHistory.data(), 
            displayCount, 
            mRewardHistory.writeIdx,
            nullptr, 
            -1.0f, 1.0f, 
            ImVec2(310, 50));
        
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "DAMAGE TAKEN");
        ImGui::PlotLines("##damage_taken", 
            mRewardHistory.damageTakenHistory.data(), 
            displayCount, 
            mRewardHistory.writeIdx,
            nullptr, 
            -1.0f, 1.0f, 
            ImVec2(310, 50));
        
        ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "AIRTIME");
        ImGui::PlotLines("##airtime", 
            mRewardHistory.airtimeHistory.data(), 
            displayCount, 
            mRewardHistory.writeIdx,
            nullptr, 
            -1.0f, 1.0f, 
            ImVec2(310, 50));
        
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "ENERGY USED");
        ImGui::PlotLines("##energy", 
            mRewardHistory.energyHistory.data(), 
            displayCount, 
            mRewardHistory.writeIdx,
            nullptr, 
            -1.0f, 1.0f, 
            ImVec2(310, 50));
        
        ImGui::TextColored(mColorAccent, "SCALAR REWARD");
        ImGui::PlotLines("##scalar", 
            mRewardHistory.scalarHistory.data(), 
            displayCount, 
            mRewardHistory.writeIdx,
            nullptr, 
            -2.0f, 2.0f, 
            ImVec2(310, 50));
    }
    
    ImGui::End();
}

void CyberpunkUI::DrawLidarHUD(const float* lidarDistances, int numRays)
{
    ImGui::SetNextWindowPos(ImVec2(360, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(250, 250), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("LIDAR HUD", nullptr, ImGuiWindowFlags_NoCollapse);
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 center = ImGui::GetCursorScreenPos();
    center.x += 100;
    center.y += 100;
    
    float radius = 90.0f;
    
    drawList->AddCircle(center, radius, IM_COL32(0, 255, 218, 80), 64, 1.0f);
    drawList->AddCircle(center, radius * 0.5f, IM_COL32(0, 255, 218, 40), 64, 1.0f);
    drawList->AddLine(
        ImVec2(center.x - radius, center.y), 
        ImVec2(center.x + radius, center.y), 
        IM_COL32(0, 255, 218, 40), 1.0f);
    drawList->AddLine(
        ImVec2(center.x, center.y - radius), 
        ImVec2(center.x, center.y + radius), 
        IM_COL32(0, 255, 218, 40), 1.0f);
    
    for (int i = 0; i < numRays; ++i)
    {
        float angle = (2.0f * 3.14159f * i) / numRays;
        float dist = lidarDistances[i] / 20.0f;
        dist = std::clamp(dist, 0.0f, 1.0f);
        
        float hitRadius = radius * dist;
        ImVec2 end(
            center.x + cosf(angle) * hitRadius,
            center.y + sinf(angle) * hitRadius
        );
        
        int intensity = static_cast<int>(255 * (1.0f - dist));
        drawList->AddLine(center, end, IM_COL32(255, intensity, 0, 200), 2.0f);
        drawList->AddCircleFilled(end, 3.0f, IM_COL32(255, 0, 128, 255));
    }
    
    drawList->AddCircleFilled(center, 4.0f, IM_COL32(0, 255, 218, 255));
    
    ImGui::End();
}
